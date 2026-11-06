"""Training job backend interface and the caller-scoped CodeFlare implementation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import shlex
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

import anyio
from codeflare_sdk import RayJob
from codeflare_sdk.ray.cluster.config import ClusterConfiguration
from fastapi import Request
from jsonschema import ValidationError
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from pydantic import ValidationError as ModelValidationError

from training_service import catalog
from training_service.auth import CallerIdentity
from training_service.contract import component_validator
from training_service.dataset_preflight import validate_dataset_columns, validate_dataset_location
from training_service.errors import ApiError
from training_service.hf_dataset import validate_hf_dataset, validate_hf_model
from training_service.kubernetes_client import (
    TIMEOUT,
    caller_client,
    caller_configuration,
    describe_not_found,
    validate_names,
)
from training_service.logs import CodeFlareLogClient, LogClient, logs_not_found, logs_unavailable
from training_service.ray_entrypoint import get_backend
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
_GPU_TYPES = frozenset({"nvidia.com/gpu", "amd.com/gpu", "intel.com/gpu"})
_RAYJOB_CREATE_PATH = "/apis/{group}/{version}/namespaces/{namespace}/{plural}"


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
        # CodeFlare's client verification and queue lookup also use this transport.
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
        """Submit a caller-authorized training job and return its identifier."""


class JobsBackend(JobSubmitter, Protocol):
    """Operations used by API handlers; backend resource formats stay private."""

    backend_name: str
    api_group: str
    resource: str

    def list_jobs(
        self,
        request: Request,
        project: str,
        status: str | None,
        queue: str | None,
        limit: int | None,
        page_token: str | None,
    ) -> dict[str, Any]: ...

    def get_job(self, request: Request, project: str, name: str) -> dict[str, Any]: ...

    def delete_job(self, request: Request, project: str, name: str) -> None: ...

    def log_client(self, request: Request, project: str, name: str) -> tuple[LogClient, str]: ...


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


def validate_algorithm_parameters(
    name: str, parameters: dict[str, Any] | None, runtime_image: str | None = None
) -> dict[str, Any]:
    """Use the startup catalog for schema validation and retain execution-profile checks."""
    values = dict(parameters or {})
    catalog.validate_parameters(name, values, "ray", runtime_image)
    try:
        backend = get_backend(name)
        selected = values.pop("backend", backend.backend_name)
        if selected != backend.backend_name:
            raise ValueError(
                f"Algorithm '{name}' currently requires Training Hub backend "
                f"'{backend.backend_name}' for its execution profile."
            )
        return backend.validate_parameters(values)
    except ValueError as error:
        raise _invalid(str(error)) from error


def _validate_request(
    request: CreateJobRequest, runtime_image: str | None = None
) -> tuple[ResourceSpec, dict[str, Any]]:
    if request.backend not in (None, JobBackend.RAY):
        raise ApiError(501, "not_implemented", "Only the Ray backend is implemented.")
    parameters = validate_algorithm_parameters(
        request.algorithm.name, request.algorithm.parameters, runtime_image
    )
    if request.queue_name is not None and (
        len(request.queue_name) > 63 or not _NAME_PATTERN.fullmatch(request.queue_name)
    ):
        raise _invalid("queue_name must be a DNS-1123 label.")
    resources = request.resources or ResourceSpec()
    cpu = resources.cpus_per_node or "4"
    memory = resources.memory_per_node or "16Gi"
    if not _CPU_PATTERN.fullmatch(cpu) or not _MEMORY_PATTERN.fullmatch(memory):
        raise _invalid("CPU or memory resource quantity is invalid.")
    if resources.gpu_type is not None and resources.gpu_type not in _GPU_TYPES:
        raise _invalid("gpu_type must be a supported GPU resource name.")
    gpu_count = resources.gpus_per_node if resources.gpus_per_node is not None else 1
    try:
        get_backend(request.algorithm.name).validate_resources(
            gpu_count, resources.num_nodes or 1, resources.gpu_type or "nvidia.com/gpu", parameters
        )
    except ValueError as error:
        raise _invalid(str(error)) from error
    return resources, parameters


def _caller_api_client(token: str) -> client.ApiClient:
    return _CallerApiClient(caller_configuration(f"Bearer {token}"))


class _MetadataRayJob(RayJob):
    """Add required API metadata to the SDK-built resource before SDK submission."""

    def __init__(self, *, annotations: Mapping[str, str], **kwargs: Any) -> None:
        self._submission_annotations = dict(annotations)
        super().__init__(**kwargs)

    def _build_rayjob_cr(self) -> dict[str, Any]:
        resource: dict[str, Any] = super()._build_rayjob_cr()
        resource["metadata"].setdefault("annotations", {}).update(self._submission_annotations)
        return resource


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
) -> tuple[
    dict[str, dict[str, str]], list[client.V1Volume], list[client.V1VolumeMount], dict[str, str]
]:
    """Resolve each reference with the caller's Kubernetes client."""
    volumes: list[client.V1Volume] = []
    mounts: list[client.V1VolumeMount] = []
    credential_dirs: dict[str, str] = {}
    credential_schemes: dict[str, str] = {}
    token_keys: dict[str, str] = {}
    pvc_paths: dict[str, str] = {}
    hf_tokens: dict[str, str] = {}
    for location in locations.values():
        name = location["name"]
        scheme = location["scheme"]
        if scheme == "pvc" and name not in pvc_paths:
            with describe_not_found("PersistentVolumeClaim", name, project=project):
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
            with describe_not_found("Secret", name, project=project):
                secret = api.read_namespaced_secret(
                    name, project, _request_timeout=_K8S_TIMEOUT_SECONDS
                )
            keys = set(secret.data or {}) | set(secret.string_data or {})
            if scheme == "hf":
                token_key = "HF_TOKEN" if "HF_TOKEN" in keys else "token"
                if token_key not in keys:
                    raise _invalid("Hugging Face Secret must contain HF_TOKEN or token.")
                raw = (secret.string_data or {}).get(token_key)
                if raw is None:
                    try:
                        raw = base64.b64decode(
                            (secret.data or {})[token_key], validate=True
                        ).decode("utf-8")
                    except (binascii.Error, UnicodeDecodeError, KeyError, ValueError) as error:
                        raise _invalid("Hugging Face Secret token is invalid.") from error
                if not raw:
                    raise _invalid("Hugging Face Secret token is empty.")
                hf_tokens[name] = raw
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
    return resolved, volumes, mounts, hf_tokens


def _map_backend_error(error: Exception, project: str, name: str) -> ApiError:
    if isinstance(error, _SdkSubmissionError):
        error = error.error
    if isinstance(error, ApiError):
        return error
    target = f"training job '{name}' in project '{project}'"
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
                403, "forbidden", f"Caller cannot access a resource required to submit {target}."
            )
        if status == 404:
            return ApiError(
                404, "not_found", f"A resource required to submit {target} was not found."
            )
        if status == 409:
            return ApiError(
                409,
                "already_exists",
                f"Training job '{name}' already exists in project '{project}'.",
            )
        if status in {400, 422}:
            return _invalid(f"The backend rejected the configuration for {target}.")
        return ApiError(503, "submission_unavailable", "RayJob submission is unavailable.")
    if isinstance(error, ValueError):
        if "Cluster config image:" in str(error):
            return ApiError(503, "submission_unavailable", "Ray training image is incompatible.")
        return _invalid("The submitted job configuration is invalid.")
    return ApiError(503, "submission_unavailable", "RayJob submission is unavailable.")


class CodeFlareJobBackend:
    """Keep Ray resource handling behind the backend-neutral job interface."""

    backend_name = "ray"
    api_group = "ray.io"
    resource = "rayjobs"

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
            raise _map_backend_error(error, project, request.name) from error

    def _submit_sync(
        self, project: str, request: CreateJobRequest, caller: CallerIdentity, token: str
    ) -> str:
        if not self._image:
            raise ApiError(503, "submission_unavailable", "A Ray training image is not configured.")
        resources, parameters = _validate_request(request, self._image)
        locations = {
            "model": _parse_location(request.model.uri.root),
            "dataset": _parse_location(request.dataset.uri.root),
        }
        dataset_options = request.dataset.model_dump(exclude={"uri"}, exclude_none=True)
        if dataset_options:
            if locations["dataset"]["scheme"] != "hf":
                raise _invalid(
                    "dataset.config and dataset.split require a Hugging Face dataset URI."
                )
            if get_backend(request.algorithm.name).remote_hf:
                raise _invalid(
                    "dataset.config and dataset.split are not supported by the remote GRPO loader."
                )
            locations["dataset"].update(dataset_options)
        if request.output is not None:
            locations["output"] = _parse_location(request.output.destination_uri, output=True)
        backend = get_backend(request.algorithm.name)
        try:
            backend.validate_locations(locations, parameters)
        except ValueError as error:
            raise _invalid(str(error)) from error
        validate_dataset_location(locations["dataset"])
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
            resolved, volumes, mounts, hf_tokens = _storage_plan(
                core_api, project, locations, resources.num_nodes or 1
            )
            if locations["model"]["scheme"] == "hf":
                validate_hf_model(
                    locations["model"]["repository"], hf_tokens[locations["model"]["name"]]
                )
            if locations["dataset"]["scheme"] == "hf":
                columns = validate_hf_dataset(
                    locations["dataset"]["repository"],
                    locations["dataset"].get("config"),
                    locations["dataset"].get("split", "train"),
                    hf_tokens[locations["dataset"]["name"]],
                    remote_loader=backend.remote_hf,
                )
                validate_dataset_columns(request.algorithm.name, parameters, columns)
            gpu_type = resources.gpu_type or "nvidia.com/gpu"
            gpu_count = resources.gpus_per_node if resources.gpus_per_node is not None else 1
            cpu = resources.cpus_per_node or "4"
            memory = resources.memory_per_node or "16Gi"
            workers = (resources.num_nodes or 1) - 1
            layout = backend.cluster_options(gpu_count, cpu, memory, resolved)
            cluster = ClusterConfiguration(
                image=self._image,
                num_workers=workers,
                head_cpu_requests=layout["head_cpu"],
                head_cpu_limits=layout["head_cpu"],
                head_memory_requests=layout["head_memory"],
                head_memory_limits=layout["head_memory"],
                head_extended_resource_requests={gpu_type: layout["head_gpu_count"]}
                if layout["head_gpu_count"]
                else {},
                worker_cpu_requests=cpu,
                worker_cpu_limits=cpu,
                worker_memory_requests=memory,
                worker_memory_limits=memory,
                worker_extended_resource_requests={gpu_type: gpu_count} if gpu_count else {},
                envs=layout["envs"],
                volumes=volumes,
                volume_mounts=mounts,
            )
            job = _MetadataRayJob(
                job_name=request.name,
                namespace=project,
                entrypoint=_entrypoint(
                    {
                        **resolved,
                        "algorithm": request.algorithm.name,
                        "parameters": parameters,
                        "gpus_per_node": gpu_count,
                        "num_workers": workers,
                    }
                ),
                cluster_config=cluster,
                local_queue=request.queue_name,
                annotations=annotations,
                api_client=api_client,
            )
            identifier = str(job.submit())
            if identifier != request.name:
                raise RuntimeError("CodeFlare returned an unexpected RayJob identifier")
            return identifier

    def list_jobs(
        self,
        request: Request,
        project: str,
        status: str | None,
        queue: str | None,
        limit: int | None,
        page_token: str | None,
    ) -> dict[str, Any]:
        validate_names(project)
        scope = _scope(project, status, queue)
        after = _cursor(page_token, scope)
        items = []
        with caller_client(request) as api:
            # Kubernetes list on a nonexistent namespace may otherwise look empty.
            with describe_not_found("Project", project):
                client.CoreV1Api(api).read_namespace(project, _request_timeout=TIMEOUT)
            custom = client.CustomObjectsApi(api)
            continuation = ""
            while True:
                page = custom.list_namespaced_custom_object(
                    "ray.io",
                    "v1",
                    project,
                    "rayjobs",
                    limit=500,
                    _continue=continuation,
                    _request_timeout=TIMEOUT,
                )
                for job in page["items"]:
                    item = summary(job)
                    if (status is None or item["status"] == status) and (
                        queue is None or item.get("queue_name") == queue
                    ):
                        items.append(item)
                continuation = page.get("metadata", {}).get("continue", "")
                if not continuation:
                    break
        items.sort(key=lambda item: item["id"])
        remaining = [item for item in items if item["id"] > after]
        selected = remaining[: limit or 20]
        response: dict[str, Any] = {"items": selected, "total_count": len(items)}
        if len(remaining) > len(selected):
            marker = {"scope": scope, "after": selected[-1]["id"]}
            response["next_page_token"] = base64.urlsafe_b64encode(
                json.dumps(marker).encode()
            ).decode()
        return response

    def get_job(self, request: Request, project: str, name: str) -> dict[str, Any]:
        validate_names(project, name)
        with caller_client(request) as api:
            return detail(read_job(api, project, name), project)

    def delete_job(self, request: Request, project: str, name: str) -> None:
        validate_names(project, name)
        with caller_client(request) as api:
            job = read_job(api, project, name)
            state = job.get("status", {})
            # Use the execution status, not a derived infrastructure-failure label.
            # A retrying deployment can still launch work after an earlier attempt ended.
            if state.get("jobStatus") not in TERMINAL or state.get("jobDeploymentStatus") in {
                "Retrying",
                "Initializing",
                "Waiting",
                "Suspending",
                "Suspended",
            }:
                raise ApiError(409, "job_not_terminal", "Only terminal RayJobs can be deleted.")
            if job.get("spec", {}).get("clusterSelector"):
                raise ApiError(
                    409, "shared_cluster", "Shared RayClusters cannot be deleted by this API."
                )
            try:
                owned_cluster(api, project, job)
            except ApiException as error:
                if error.status != 404:
                    raise
                # Normal when shutdown-after-finish already removed the owned cluster.
            metadata = job["metadata"]
            with describe_not_found("Training job", name, project=project):
                client.CustomObjectsApi(api).delete_namespaced_custom_object(
                    "ray.io",
                    "v1",
                    project,
                    "rayjobs",
                    name,
                    body=client.V1DeleteOptions(
                        propagation_policy="Foreground",
                        preconditions=client.V1Preconditions(
                            uid=metadata["uid"],
                            resource_version=metadata["resourceVersion"],
                        ),
                    ),
                    _request_timeout=TIMEOUT,
                )

    def log_client(self, request: Request, project: str, name: str) -> tuple[LogClient, str]:
        validate_names(project, name)
        with caller_client(request) as api:
            job = read_job(api, project, name)
            state = job.get("status", {})
            submission_id = state.get("jobId")
            # KubeRay allocates IDs before creating the cluster or submitting
            # to Ray. Do not report its not-yet-created cluster as gone.
            if state.get("jobStatus") not in _STATUSES:
                if job_status(job) == "PENDING":
                    raise ApiError(
                        409, "logs_not_ready", "Driver logs are not available yet; retry later."
                    )
                raise logs_not_found(project, name)
            try:
                cluster = owned_cluster(api, project, job)
            except ApiException as error:
                if error.status == 404:
                    raise ApiError(
                        410,
                        "ray_cluster_gone",
                        "Driver logs are gone because the RayCluster was removed.",
                    ) from None
                raise
            except ApiError as error:
                if error.code == "unowned_cluster":
                    raise ApiError(
                        403, "forbidden", "RayCluster ownership could not be verified."
                    ) from None
                raise
        if not cluster or not submission_id:
            if job_status(job) == "PENDING":
                raise ApiError(
                    409, "logs_not_ready", "Driver logs are not available yet; retry later."
                )
            raise logs_not_found(project, name)
        if (
            not isinstance(submission_id, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,1024}", submission_id) is None
        ):
            raise logs_unavailable()
        gateway = os.getenv("TRAINING_SERVICE_RAY_GATEWAY_URL", "").rstrip("/")
        try:
            parsed = urlsplit(gateway)
        except ValueError:
            raise logs_unavailable() from None
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or os.getenv("RAY_ADDRESS")
            or os.getenv("RAY_API_SERVER_ADDRESS")
        ):
            raise logs_unavailable()
        try:
            sdk = CodeFlareLogClient(
                f"{gateway}/ray/{project}/{cluster}",
                headers={"Authorization": request.headers["authorization"]},
                verify=os.getenv("TRAINING_SERVICE_RAY_CA_FILE") or True,
            )
        except ApiError:
            raise
        except Exception:
            raise logs_unavailable() from None
        return sdk, submission_id


TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
_STATUSES = TERMINAL | {"PENDING", "RUNNING"}
_ANNOTATION = "training.rhoai.io/"


def read_job(api: client.ApiClient, project: str, name: str) -> dict[str, Any]:
    """Resolve only the requested namespace, using the caller's client."""
    with describe_not_found("Training job", name, project=project):
        result: dict[str, Any] = client.CustomObjectsApi(api).get_namespaced_custom_object(
            "ray.io", "v1", project, "rayjobs", name, _request_timeout=TIMEOUT
        )
    return result


def job_status(job: dict[str, Any]) -> str:
    """Distinguish infrastructure failure from pending/waiting provisioning."""
    state = job.get("status", {})
    if state.get("jobDeploymentStatus") in {"Failed", "ValidationFailed"}:
        return "FAILED"
    status = state.get("jobStatus", "PENDING")
    return status if status in _STATUSES else "PENDING"


def summary(job: dict[str, Any]) -> dict[str, Any]:
    metadata = job["metadata"]
    result = {
        "id": metadata["name"],
        "status": job_status(job),
        "created_at": metadata["creationTimestamp"],
    }
    if queue := metadata.get("labels", {}).get("kueue.x-k8s.io/queue-name"):
        result["queue_name"] = queue
    return result


def _timestamp(value: Any) -> str | None:
    if isinstance(value, str) and value and not value.startswith("0001-"):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        try:
            return datetime.fromtimestamp(value / 1000, UTC).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    return None


def detail(job: dict[str, Any], project: str) -> dict[str, Any]:
    """Expose allowlisted fields, never raw runtime environments or Secret values."""
    result = summary(job)
    result.update(name=result["id"], project=project, selected_backend="ray")
    annotations = job["metadata"].get("annotations", {})
    if submitter := annotations.get(_ANNOTATION + "submitted-by"):
        result["submitted_by"] = submitter
    submission = annotations.get(_ANNOTATION + "submission")
    if isinstance(submission, str) and len(submission.encode()) <= 65536:
        try:
            accepted = CreateJobRequest.model_validate_json(submission)
            result["submission"] = accepted.model_dump(mode="json", exclude_unset=True)
        except (ModelValidationError, ValueError):
            # External jobs may have missing, legacy or malformed API annotations.
            pass
    state = job.get("status", {})
    for source, target in (("startTime", "started_at"), ("endTime", "finished_at")):
        if source == "endTime" and result["status"] not in TERMINAL:
            continue
        if timestamp := _timestamp(state.get(source)):
            result[target] = timestamp
    if result["status"] == "FAILED" and state.get("message"):
        result["failure_reason"] = state["message"]
    if resources := _resources(job.get("spec", {}).get("rayClusterSpec", {})):
        result["resources"] = resources
    return result


def _resources(cluster: dict[str, Any]) -> dict[str, Any] | None:
    """Report homogeneous Ray nodes, including the submission adapter's head."""
    groups = cluster.get("workerGroupSpecs", [])
    if cluster.get("headGroupSpec"):
        groups = [dict(cluster["headGroupSpec"], replicas=1), *groups]
    groups = [group for group in groups if group.get("replicas") != 0]
    if not groups:
        return None
    node_requests = []
    for group in groups:
        containers = group.get("template", {}).get("spec", {}).get("containers", [])
        if not containers:
            return None
        node_requests.append(
            {
                key: value
                for key, value in containers[0].get("resources", {}).get("requests", {}).items()
                if key in {"cpu", "memory"} or key.endswith("/gpu")
            }
        )
    requests = node_requests[0]
    if any(item != requests for item in node_requests[1:]):
        return None
    gpu_keys = [key for key in requests if key.endswith("/gpu")]
    if len(gpu_keys) > 1:
        return None
    result: dict[str, Any] = {}
    if all(type(group.get("replicas")) is int and group["replicas"] > 0 for group in groups):
        result["num_nodes"] = sum(group["replicas"] for group in groups)
    for source, target in (("cpu", "cpus_per_node"), ("memory", "memory_per_node")):
        if source in requests:
            result[target] = str(requests[source])
    if gpu_keys:
        try:
            result.update(gpu_type=gpu_keys[0], gpus_per_node=int(requests[gpu_keys[0]]))
        except (ValueError, TypeError):
            return None
    try:
        ResourceSpec.model_validate(result)
    except ModelValidationError:
        return None
    return result or None


def _scope(project: str, status: str | None, queue: str | None) -> str:
    return hashlib.sha256(json.dumps([project, status, queue]).encode()).hexdigest()


def _cursor(token: str | None, scope: str) -> str:
    if token is None:
        return ""
    try:
        if len(token) > 2048:
            raise ValueError
        data = json.loads(base64.b64decode(token, altchars=b"-_", validate=True))
        if not isinstance(data, dict) or data.get("scope") != scope:
            raise ValueError
        after = data["after"]
        if not isinstance(after, str) or not after:
            raise ValueError
        return after
    except (ValueError, TypeError, KeyError, binascii.Error):
        raise ApiError(
            400, "invalid_page_token", "Invalid continuation token for these filters."
        ) from None


def owned_cluster(api: client.ApiClient, project: str, job: dict[str, Any]) -> str | None:
    """Fail closed rather than act on a shared or caller-injected Ray endpoint."""
    name = job.get("status", {}).get("rayClusterName")
    if not name:
        return None
    validate_names(project, name)
    cluster = client.CustomObjectsApi(api).get_namespaced_custom_object(
        "ray.io", "v1", project, "rayclusters", name, _request_timeout=TIMEOUT
    )
    if not any(
        owner.get("uid") == job["metadata"]["uid"]
        and owner.get("name") == job["metadata"]["name"]
        and owner.get("kind") == "RayJob"
        and owner.get("apiVersion") == "ray.io/v1"
        and owner.get("controller") is True
        for owner in cluster["metadata"].get("ownerReferences", [])
    ):
        raise ApiError(
            409, "unowned_cluster", "RayCluster is not exclusively owned by this RayJob."
        )
    return str(name)
