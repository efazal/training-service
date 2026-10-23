"""Caller-scoped CodeFlare submission for the basic Training API request."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

import anyio
from codeflare_sdk import RayJob
from codeflare_sdk.common.kubernetes_cluster import auth as codeflare_auth
from codeflare_sdk.ray.rayjobs.config import ManagedClusterConfig
from jsonschema import ValidationError
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from training_service.auth import CallerIdentity
from training_service.contract import component_validator
from training_service.errors import ApiError
from training_service_api.models.create_job_request import CreateJobRequest
from training_service_api.models.job_backend import JobBackend
from training_service_api.models.resource_spec import ResourceSpec

_LOG = logging.getLogger(__name__)
_K8S_TIMEOUT_SECONDS = 10
_ANNOTATION_SUBMISSION = "training.rhoai.io/submission"
_ANNOTATION_SUBMITTER = "training.rhoai.io/submitted-by"
_NAME_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_CPU_PATTERN = re.compile(r"^(?:[1-9][0-9]*|[1-9][0-9]*m|0\.0*[1-9][0-9]*)$")
_MEMORY_PATTERN = re.compile(r"^[1-9][0-9]*(?:Ki|Mi|Gi|Ti|K|M|G|T)$")
_BASIC_PARAMETERS = frozenset(
    {"learning_rate", "num_epochs", "effective_batch_size", "max_seq_len"}
)
_GPU_TYPES = frozenset({"nvidia.com/gpu", "amd.com/gpu", "intel.com/gpu"})
_RAYJOB_CREATE_PATH = "/apis/{group}/{version}/namespaces/{namespace}/{plural}"
_SDK_AUTH_LOCK = Lock()


class _SdkSubmissionError(Exception):
    """Carry a RayJob create error past the SDK transport without exposing its body."""

    def __init__(self, error: ApiException) -> None:
        super().__init__("The SDK RayJob create request failed.")
        self.error = error


class _CallerApiClient(client.ApiClient):
    """Keep the caller identity and preserve errors from SDK-owned RayJob creation."""

    def call_api(self, resource_path: str, method: str, *args: Any, **kwargs: Any) -> Any:
        path_params = args[0] if args else kwargs.get("path_params")
        is_rayjob_create = (
            resource_path == _RAYJOB_CREATE_PATH
            and method == "POST"
            and isinstance(path_params, dict)
            and path_params.get("group") == "ray.io"
            and path_params.get("version") == "v1"
            and path_params.get("plural") == "rayjobs"
        )
        # CodeFlare's client verification and queue lookup also use this
        # transport while holding the process-wide SDK auth lock.
        if kwargs.get("_request_timeout") is None:
            kwargs["_request_timeout"] = _K8S_TIMEOUT_SECONDS
        try:
            return super().call_api(resource_path, method, *args, **kwargs)
        except ApiException as error:
            if is_rayjob_create:
                # CodeFlare's vendored RayjobApi catches ApiException and
                # returns None. Use a distinct exception type so RayJob.submit
                # remains the creator while our API can map the actual status.
                raise _SdkSubmissionError(error) from error
            raise


class JobSubmitter(Protocol):
    async def submit(
        self, project: str, request: CreateJobRequest, caller: CallerIdentity, token: str
    ) -> str:
        """Create one caller-authorized RayJob and return its resource name."""


def _invalid(message: str) -> ApiError:
    return ApiError(422, "validation_error", message)


def _parse_location(uri: str, *, output: bool = False) -> dict[str, str]:
    if output:
        try:
            component_validator("OutputSpec").validate({"destination_uri": uri})
        except ValidationError as error:
            raise _invalid("Output destination URI is invalid.") from error
    parsed = urlsplit(uri)
    scheme = parsed.scheme.lower()
    if scheme not in ({"s3", "pvc"} if output else {"hf", "s3", "pvc"}):
        raise _invalid("Unsupported storage URI scheme.")
    if not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
        raise _invalid("Storage URI must identify one project-local resource.")
    if not parsed.path.startswith("/"):
        raise _invalid("Storage URI path is invalid.")
    path = unquote(parsed.path[1:])
    if not path or path.startswith(("/", "\\")):
        raise _invalid("Storage URI path is invalid.")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise _invalid("Storage URI path contains control characters.")
    if scheme == "pvc":
        parts = path.rstrip("/").split("/")
        if any(part in {"", ".", ".."} or "\\" in part for part in parts):
            raise _invalid("PVC path must remain inside its volume.")
    if scheme == "s3":
        bucket, separator, key = path.partition("/")
        if not separator or not bucket or not key:
            raise _invalid("S3 URI requires an explicit bucket and object path.")
        return {"scheme": scheme, "name": parsed.netloc, "bucket": bucket, "path": key}
    if scheme == "hf":
        return {"scheme": scheme, "name": parsed.netloc, "repository": path}
    return {"scheme": scheme, "name": parsed.netloc, "path": path}


def _validate_request(request: CreateJobRequest) -> tuple[ResourceSpec, dict[str, int | float]]:
    if request.backend not in (None, JobBackend.RAY):
        raise ApiError(501, "not_implemented", "Only the Ray backend is implemented.")
    if request.algorithm.name != "lora_sft":
        raise _invalid("Only the basic lora_sft algorithm is supported.")
    parameters = request.algorithm.parameters or {}
    if set(parameters) - _BASIC_PARAMETERS:
        raise _invalid("Unsupported lora_sft parameter.")
    for key, value in parameters.items():
        if key == "learning_rate":
            if type(value) not in (int, float) or not (0 < value <= 1):
                raise _invalid("learning_rate must be a positive number at most 1.")
        elif type(value) is not int or value <= 0:
            raise _invalid(f"{key} must be a positive integer.")
    if request.queue_name is not None and (
        len(request.queue_name) > 63 or not _NAME_PATTERN.fullmatch(request.queue_name)
    ):
        raise _invalid("queue_name must be a DNS-1123 label.")
    resources = request.resources or ResourceSpec()
    cpu = resources.cpus_per_node or "8"
    memory = resources.memory_per_node or "32Gi"
    if not _CPU_PATTERN.fullmatch(cpu) or not _MEMORY_PATTERN.fullmatch(memory):
        raise _invalid("CPU or memory resource quantity is invalid.")
    if resources.gpu_type is not None and resources.gpu_type not in _GPU_TYPES:
        raise _invalid("gpu_type must be a supported GPU resource name.")
    return resources, parameters


def _caller_api_client(token: str) -> client.ApiClient:
    configuration = client.Configuration()
    # Never call load_incluster_config without a target Configuration: doing so
    # mutates the process-wide default. Disable its refresh hook, which would
    # otherwise silently replace this caller token with the pod's token.
    config.load_incluster_config(
        client_configuration=configuration,
        try_refresh_token=False,
    )
    configuration.api_key["authorization"] = f"Bearer {token}"
    configuration.api_key_prefix.pop("authorization", None)
    return _CallerApiClient(configuration)


class _MetadataRayJob(RayJob):
    """Add required API metadata to the SDK-built resource before SDK submission."""

    def __init__(self, *, annotations: Mapping[str, str], **kwargs: Any) -> None:
        self._submission_annotations = dict(annotations)
        super().__init__(**kwargs)

    def _build_rayjob_cr(self) -> dict[str, Any]:
        resource: dict[str, Any] = super()._build_rayjob_cr()
        resource["metadata"].setdefault("annotations", {}).update(self._submission_annotations)
        return resource


@contextmanager
def _sdk_client_scope(api_client: client.ApiClient) -> Iterator[None]:
    """Serialize SDK 0.38's process-global auth for one caller's submission."""
    with _SDK_AUTH_LOCK:
        previous_client = codeflare_auth.api_client
        previous_config_path = codeflare_auth.config_path
        try:
            # RayJob 0.38 has no api_client parameter. Its constructor and
            # submission read this global, so keep both inside one lock.
            codeflare_auth.set_api_client(api_client)
            yield
        finally:
            codeflare_auth.api_client = previous_client
            codeflare_auth.config_path = previous_config_path


def _entrypoint(configuration: dict[str, Any]) -> str:
    source = Path(__file__).with_name("ray_entrypoint.py").read_text(encoding="utf-8")
    program = source + "\nmain(" + repr(configuration) + ")\n"
    encoded = base64.b64encode(program.encode("utf-8")).decode("ascii")
    return "python -c " + shlex.quote(f'import base64; exec(base64.b64decode("{encoded}"))')


def _storage_plan(
    api: client.CoreV1Api,
    project: str,
    locations: dict[str, dict[str, str]],
    num_nodes: int,
) -> tuple[dict[str, dict[str, str]], list[client.V1Volume], list[client.V1VolumeMount]]:
    """Resolve each reference with the caller's Kubernetes client."""
    volumes: list[client.V1Volume] = []
    mounts: list[client.V1VolumeMount] = []
    credential_dirs: dict[str, str] = {}
    credential_schemes: dict[str, str] = {}
    token_keys: dict[str, str] = {}
    pvc_paths: dict[str, str] = {}
    for location in locations.values():
        name = location["name"]
        scheme = location["scheme"]
        if scheme == "pvc" and name not in pvc_paths:
            pvc = api.read_namespaced_persistent_volume_claim(
                name, project, _request_timeout=_K8S_TIMEOUT_SECONDS
            )
            if num_nodes > 1 and "ReadWriteMany" not in (pvc.spec.access_modes or []):
                raise _invalid("Multi-node training requires ReadWriteMany access for every PVC.")
            mount_path = f"/mnt/training-api/pvc/{len(pvc_paths)}"
            volume_name = f"training-pvc-{len(pvc_paths)}"
            pvc_paths[name] = mount_path
            volumes.append(
                client.V1Volume(
                    name=volume_name,
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=name
                    ),
                )
            )
            mounts.append(client.V1VolumeMount(name=volume_name, mount_path=mount_path))
        elif scheme in {"hf", "s3"} and name not in credential_dirs:
            secret = api.read_namespaced_secret(
                name, project, _request_timeout=_K8S_TIMEOUT_SECONDS
            )
            keys = set(secret.data or {}) | set(secret.string_data or {})
            if scheme == "hf":
                token_key = "HF_TOKEN" if "HF_TOKEN" in keys else "token"
                if token_key not in keys:
                    raise _invalid("Hugging Face Secret must contain HF_TOKEN or token.")
                selected = {token_key}
                token_keys[name] = token_key
            else:
                required = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"}
                if not required <= keys:
                    raise _invalid("S3 connection Secret is missing required keys.")
                selected = required | (keys & {"AWS_DEFAULT_REGION", "AWS_SESSION_TOKEN"})
            mount_path = f"/mnt/training-api/credentials/{len(credential_dirs)}"
            volume_name = f"training-secret-{len(credential_dirs)}"
            credential_dirs[name] = mount_path
            credential_schemes[name] = scheme
            volumes.append(
                client.V1Volume(
                    name=volume_name,
                    secret=client.V1SecretVolumeSource(
                        secret_name=name,
                        default_mode=0o444,
                        items=[client.V1KeyToPath(key=key, path=key) for key in sorted(selected)],
                    ),
                )
            )
            mounts.append(
                client.V1VolumeMount(name=volume_name, mount_path=mount_path, read_only=True)
            )
        elif scheme in {"hf", "s3"} and name in credential_dirs:
            if credential_schemes[name] != scheme:
                raise _invalid("One Secret cannot serve both HF and S3 locations.")

    resolved: dict[str, dict[str, str]] = {}
    for role, location in locations.items():
        item = dict(location)
        name = item.pop("name")
        if item["scheme"] == "pvc":
            item["mount_path"] = pvc_paths[name] + "/" + item.pop("path")
        else:
            item["credential_dir"] = credential_dirs[name]
            if item["scheme"] == "hf":
                item["token_key"] = token_keys[name]
        resolved[role] = item
    return resolved, volumes, mounts


def _map_backend_error(error: Exception) -> ApiError:
    if isinstance(error, _SdkSubmissionError):
        error = error.error
    if isinstance(error, ApiError):
        return error
    if isinstance(error, ApiException):
        status = error.status
        if status == 401:
            return ApiError(
                401,
                "unauthorized",
                "The OpenShift bearer token is no longer valid.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if status == 403:
            return ApiError(
                403, "forbidden", "Caller cannot access the requested project resource."
            )
        if status == 404:
            return ApiError(404, "not_found", "A referenced project resource was not found.")
        if status == 409:
            return ApiError(409, "already_exists", "A RayJob with this name already exists.")
        if status in {400, 422}:
            return _invalid("The backend rejected the submitted job configuration.")
        return ApiError(503, "submission_unavailable", "RayJob submission is unavailable.")
    if isinstance(error, ValueError):
        if "Cluster config image:" in str(error):
            return ApiError(503, "submission_unavailable", "Ray training image is incompatible.")
        return _invalid("The submitted job configuration is invalid.")
    return ApiError(503, "submission_unavailable", "RayJob submission is unavailable.")


class CodeFlareJobSubmitter:
    """Submit via CodeFlare with a short-lived, per-caller Kubernetes client."""

    def __init__(self, image: str | None = None) -> None:
        self._image = image if image is not None else os.getenv("TRAINING_RAY_IMAGE", "")
        self._limiter = anyio.CapacityLimiter(10)

    async def submit(
        self, project: str, request: CreateJobRequest, caller: CallerIdentity, token: str
    ) -> str:
        try:
            return await anyio.to_thread.run_sync(
                self._submit_sync, project, request, caller, token, limiter=self._limiter
            )
        except Exception as error:
            _LOG.warning("RayJob submission failed (%s)", type(error).__name__)
            raise _map_backend_error(error) from error

    def _submit_sync(
        self, project: str, request: CreateJobRequest, caller: CallerIdentity, token: str
    ) -> str:
        resources, parameters = _validate_request(request)
        if not self._image:
            raise ApiError(503, "submission_unavailable", "A Ray training image is not configured.")
        locations = {
            "model": _parse_location(request.model.uri.root),
            "dataset": _parse_location(request.dataset.uri.root),
        }
        if request.output is not None:
            locations["output"] = _parse_location(request.output.destination_uri, output=True)
        payload = request.model_dump(mode="json", exclude_unset=True)
        serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        if len(serialized.encode("utf-8")) > 65536:
            raise _invalid("Submission metadata exceeds the supported size.")
        annotations = {
            _ANNOTATION_SUBMISSION: serialized,
            _ANNOTATION_SUBMITTER: caller.username,
        }

        with _caller_api_client(token) as api_client:
            core_api = client.CoreV1Api(api_client)
            resolved, volumes, mounts = _storage_plan(
                core_api, project, locations, resources.num_nodes or 1
            )
            gpu_type = resources.gpu_type or "nvidia.com/gpu"
            gpu_count = resources.gpus_per_node if resources.gpus_per_node is not None else 1
            cpu = resources.cpus_per_node or "8"
            memory = resources.memory_per_node or "32Gi"
            workers = (resources.num_nodes or 1) - 1
            cluster = ManagedClusterConfig(
                image=self._image,
                num_workers=workers,
                head_cpu_requests=cpu,
                head_cpu_limits=cpu,
                head_memory_requests=memory,
                head_memory_limits=memory,
                head_accelerators={gpu_type: gpu_count} if gpu_count else {},
                worker_cpu_requests=cpu,
                worker_cpu_limits=cpu,
                worker_memory_requests=memory,
                worker_memory_limits=memory,
                worker_accelerators={gpu_type: gpu_count} if gpu_count else {},
                volumes=volumes,
                volume_mounts=mounts,
            )
            with _sdk_client_scope(api_client):
                job = _MetadataRayJob(
                    job_name=request.name,
                    namespace=project,
                    entrypoint=_entrypoint(
                        {
                            **resolved,
                            "parameters": parameters,
                        }
                    ),
                    cluster_config=cluster,
                    local_queue=request.queue_name,
                    annotations=annotations,
                )
                identifier = str(job.submit())
            if identifier != request.name:
                raise RuntimeError("CodeFlare returned an unexpected RayJob identifier")
            return identifier
