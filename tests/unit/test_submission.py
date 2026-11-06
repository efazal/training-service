"""SDK-mocked submission tests; no Ray or Kubernetes cluster is contacted."""

from __future__ import annotations

import base64
import json
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from codeflare_sdk import RayJob
from codeflare_sdk.common.kubernetes_cluster import auth as codeflare_auth
from codeflare_sdk.ray.rayjobs.config import build_ray_cluster_spec
from codeflare_sdk.vendored.python_client.kuberay_cluster_api import RayClusterApi
from codeflare_sdk.vendored.python_client.kuberay_job_api import RayjobApi
from datasets.exceptions import DatasetNotFoundError
from fastapi import Request
from fastapi.testclient import TestClient
from huggingface_hub.errors import RepositoryNotFoundError
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from training_service import catalog, hf_dataset, jobs, kubernetes_client, ray_entrypoint
from training_service import jobs as submission
from training_service.auth import CallerIdentity
from training_service.authorization import ProjectPermission
from training_service.contract import load_contract
from training_service.dataset_preflight import validate_dataset_columns, validate_dataset_location
from training_service.errors import ApiError
from training_service.jobs import validate_algorithm_parameters
from training_service.main import create_app
from training_service.ray_entrypoint import _dataset_format, get_backend
from training_service_api.models.create_job_request import CreateJobRequest
from training_service_api.models.error_response import ErrorResponse


def _sample(name: str = "pvc_inputs_output") -> CreateJobRequest:
    examples = load_contract()["paths"]["/projects/{project}/training-jobs"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]
    payload = examples[name]["value"].copy()
    payload["algorithm"] = payload["algorithm"].copy()
    if name == "pvc_inputs_output":
        # Keep test-specific overrides here, not in the generic API contract.
        payload["algorithm"]["parameters"] = {
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "effective_batch_size": 8,
            "max_seq_len": 2048,
        }
    return CreateJobRequest.model_validate(payload)


class FakeAuthenticator:
    async def authenticate(self, token: str) -> CallerIdentity | None:
        return (
            CallerIdentity("alice", "uid-alice", ("ml-team",)) if token == "caller-token" else None
        )


class FakeAuthorizer:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[ProjectPermission] = []

    async def is_allowed(
        self, identity: CallerIdentity, project: str, permission: ProjectPermission
    ) -> bool:
        assert identity.username == "alice"
        assert project == "team-alpha"
        self.calls.append(permission)
        return self.allowed


class RecordingSubmitter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, CreateJobRequest, CallerIdentity, str]] = []

    async def submit(
        self, project: str, request: CreateJobRequest, caller: CallerIdentity, token: str
    ) -> str:
        self.calls.append((project, request, caller, token))
        return request.name


def test_post_authorizes_then_submits_with_verified_caller_and_202() -> None:
    authorizer = FakeAuthorizer()
    submitter = RecordingSubmitter()
    response = TestClient(create_app(FakeAuthenticator(), authorizer, submitter)).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=_sample().model_dump(mode="json", exclude_unset=True),
        headers={"Authorization": "Bearer caller-token", "X-Remote-User": "mallory"},
    )
    assert response.status_code == 202
    assert response.headers["location"] == (
        "/api/v1/projects/team-alpha/training-jobs/pvc-training"
    )
    assert response.json() == {
        "id": "pvc-training",
        "name": "pvc-training",
        "project": "team-alpha",
        "queue_name": "team-alpha-local-queue",
        "status": "SUBMITTED",
        "selected_backend": "ray",
    }
    assert authorizer.calls == [ProjectPermission("ray.io", "rayjobs", "create")]
    assert submitter.calls == [
        (
            "team-alpha",
            _sample(),
            CallerIdentity("alice", "uid-alice", ("ml-team",)),
            "caller-token",
        )
    ]


def test_omitted_queue_is_not_invented_in_receipt() -> None:
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    del payload["queue_name"]
    response = TestClient(
        create_app(FakeAuthenticator(), FakeAuthorizer(), RecordingSubmitter())
    ).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=payload,
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 202
    assert "queue_name" not in response.json()


@pytest.mark.parametrize(
    "change,expected",
    [
        ("missing_name", "Missing required field: name."),
        ("extra_model_field", "Unsupported field: model.secret_ref."),
        ("wrong_resource_type", "resources.num_nodes must be an integer."),
        (
            "unknown_parameter",
            "Unsupported parameter(s) lora_rank for algorithm 'lora_sft' in the configured "
            "training runtime. See GET /api/v1/algorithms/lora_sft.",
        ),
    ],
)
def test_submission_errors_identify_safe_field_and_reason(change: str, expected: str) -> None:
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    if change == "missing_name":
        del payload["name"]
    elif change == "extra_model_field":
        payload["model"]["secret_ref"] = "do-not-echo-this-value"
    elif change == "wrong_resource_type":
        payload["resources"]["num_nodes"] = "two"
    else:
        payload["algorithm"]["parameters"]["lora_rank"] = 8
    submitter = (
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")
        if change == "unknown_parameter"
        else RecordingSubmitter()
    )
    response = TestClient(create_app(FakeAuthenticator(), FakeAuthorizer(), submitter)).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=payload,
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 422
    assert response.json()["message"] == expected
    assert "do-not-echo-this-value" not in response.text


@pytest.mark.parametrize(
    "algorithm,parameters,expected",
    [
        ("osft", {}, "Missing required parameter(s):"),
        ("lora_sft", {"num_epochs": "three"}, "Parameter 'num_epochs' must be integer or null."),
        ("lora_sft", {"trust_remote_code": True}, "must match the runtime's allowed value(s)"),
    ],
)
def test_algorithm_parameter_errors_explain_the_constraint(
    algorithm: str, parameters: dict[str, Any], expected: str
) -> None:
    with pytest.raises(ApiError) as raised:
        validate_algorithm_parameters(algorithm, parameters)
    assert raised.value.status_code == 422
    assert expected in raised.value.message
    assert f"GET /api/v1/algorithms/{algorithm}" in raised.value.message


def test_post_denied_or_unauthenticated_never_calls_submitter() -> None:
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    submitter = RecordingSubmitter()
    denied = TestClient(create_app(FakeAuthenticator(), FakeAuthorizer(False), submitter)).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=payload,
        headers={"Authorization": "Bearer caller-token"},
    )
    assert denied.status_code == 403
    unauthenticated = TestClient(create_app(FakeAuthenticator(), FakeAuthorizer(), submitter)).post(
        "/api/v1/projects/team-alpha/training-jobs", json=payload
    )
    assert unauthenticated.status_code == 401
    assert submitter.calls == []


@pytest.mark.parametrize(
    "body,content_type,status_code,code",
    [
        ('{"name":', "application/json", 400, "bad_request"),
        ("not-json", "text/plain", 415, "unsupported_media_type"),
        ("{}", "application/json", 422, "validation_error"),
    ],
)
def test_invalid_submission_uses_documented_error_envelope(
    body: str, content_type: str, status_code: int, code: str
) -> None:
    submitter = RecordingSubmitter()
    response = TestClient(create_app(FakeAuthenticator(), FakeAuthorizer(), submitter)).post(
        "/api/v1/projects/team-alpha/training-jobs",
        content=body,
        headers={
            "Authorization": "Bearer caller-token",
            "Content-Type": content_type,
        },
    )
    assert response.status_code == status_code
    assert response.json()["code"] == code
    ErrorResponse.model_validate(response.json())
    assert "message" in response.json()
    assert submitter.calls == []


@pytest.mark.parametrize("advanced_field", ["secret_ref", "lora_rank"])
def test_unrecognized_credential_and_algorithm_fields_are_rejected(
    advanced_field: str,
) -> None:
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    if advanced_field == "secret_ref":
        payload["model"]["secret_ref"] = "another-secret"
    else:
        payload["algorithm"]["parameters"]["lora_rank"] = 8

    response = TestClient(
        create_app(
            FakeAuthenticator(),
            FakeAuthorizer(),
            submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    ).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=payload,
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    ErrorResponse.model_validate(response.json())


def test_missing_training_image_returns_503_without_kubernetes_calls() -> None:
    response = TestClient(
        create_app(
            FakeAuthenticator(),
            FakeAuthorizer(),
            submission.CodeFlareJobBackend(""),
        )
    ).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=_sample().model_dump(mode="json", exclude_unset=True),
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "submission_unavailable"
    ErrorResponse.model_validate(response.json())


def test_reads_dispatch_without_invoking_submitter(monkeypatch: pytest.MonkeyPatch) -> None:
    submitter = RecordingSubmitter()
    calls = []

    def list_jobs(self: Any, request: Any, project: str, *args: Any) -> dict[str, Any]:
        calls.append((request.state.caller_identity.username, project))
        return {"items": [], "total_count": 0}

    monkeypatch.setattr(jobs.CodeFlareJobBackend, "list_jobs", list_jobs)
    response = TestClient(create_app(FakeAuthenticator(), FakeAuthorizer(), submitter)).get(
        "/api/v1/projects/team-alpha/training-jobs",
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 200
    assert response.json() == {"items": [], "total_count": 0}
    assert calls == [("alice", "team-alpha")]
    assert submitter.calls == []


class FakeApiClient:
    def __init__(self, token: str) -> None:
        self.token = token

    def __enter__(self) -> FakeApiClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeCoreApi:
    def __init__(self) -> None:
        self.pvc_reads: list[tuple[str, str]] = []
        self.secret_reads: list[tuple[str, str]] = []
        self.pvc_access_modes: dict[str, list[str]] = {}

    def read_namespaced_persistent_volume_claim(
        self, name: str, namespace: str, **_kwargs: Any
    ) -> object:
        self.pvc_reads.append((name, namespace))
        return SimpleNamespace(
            spec=SimpleNamespace(access_modes=self.pvc_access_modes.get(name, ["ReadWriteMany"]))
        )

    def read_namespaced_secret(self, name: str, namespace: str, **_kwargs: Any) -> SimpleNamespace:
        self.secret_reads.append((name, namespace))
        if name == "hf-secret-token":
            keys = {"HF_TOKEN"}
        else:
            keys = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"}
        values = {key: base64.b64encode(b"secret-sentinel").decode() for key in keys}
        return SimpleNamespace(data=values, string_data=None)


class FakeRayJob:
    instances: list[FakeRayJob] = []
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.instances.append(self)

    def submit(self) -> str:
        self.client_at_submit = self.kwargs["api_client"]
        if self.error is not None:
            raise self.error
        return str(self.kwargs["job_name"])


def _mock_sdk(monkeypatch: pytest.MonkeyPatch) -> FakeCoreApi:
    core = FakeCoreApi()
    FakeRayJob.instances = []
    FakeRayJob.error = None
    monkeypatch.setattr(submission, "_caller_api_client", FakeApiClient)
    monkeypatch.setattr(submission.client, "CoreV1Api", lambda api_client: core)
    monkeypatch.setattr(submission, "_MetadataRayJob", FakeRayJob)
    monkeypatch.setattr(submission, "validate_hf_model", lambda *_args: None)
    monkeypatch.setattr(submission, "validate_hf_dataset", lambda *_args, **_kwargs: None)
    return core


def test_pvc_submission_maps_resources_and_retains_only_safe_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _mock_sdk(monkeypatch)
    body = _sample()
    identifier = submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        "team-alpha", body, CallerIdentity("alice", None, ()), "caller-token"
    )
    assert identifier == body.name
    assert core.pvc_reads == [
        ("model-storage", "team-alpha"),
        ("training-data", "team-alpha"),
        ("training-output", "team-alpha"),
    ]
    kwargs = FakeRayJob.instances[0].kwargs
    assert kwargs["namespace"] == "team-alpha"
    assert kwargs["local_queue"] == body.queue_name
    assert kwargs["api_client"].token == "caller-token"
    assert FakeRayJob.instances[0].client_at_submit.token == "caller-token"
    cluster = kwargs["cluster_config"]
    assert cluster.image == "quay.io/example/ray:2.58.0"
    assert cluster.num_workers == 0
    assert cluster.head_extended_resource_requests == {"nvidia.com/gpu": 1}
    assert cluster.worker_extended_resource_requests == {"nvidia.com/gpu": 1}
    assert len(cluster.volumes) == 3
    assert kwargs["annotations"]["training.rhoai.io/submitted-by"] == "alice"
    assert json.loads(kwargs["annotations"]["training.rhoai.io/submission"]) == body.model_dump(
        mode="json", exclude_unset=True
    )
    assert "caller-token" not in str(kwargs["annotations"])
    assert "caller-token" not in kwargs["entrypoint"]


def test_multinode_pvc_requires_read_write_many(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _mock_sdk(monkeypatch)
    core.pvc_access_modes["training-data"] = ["ReadWriteOnce"]
    body = _sample()
    body.algorithm.name = "lora_grpo"
    body.algorithm.parameters = {}
    body.resources.num_nodes = 2
    submitter = submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")

    with pytest.raises(ApiError) as raised:
        submitter._submit_sync("team-alpha", body, CallerIdentity("alice", None, ()), "token")
    assert raised.value.status_code == 422
    assert raised.value.code == "validation_error"
    assert FakeRayJob.instances == []

    core.pvc_access_modes["training-data"] = ["ReadWriteMany"]
    assert (
        submitter._submit_sync("team-alpha", body, CallerIdentity("alice", None, ()), "token")
        == body.name
    )


@pytest.mark.parametrize("cpu", ["0.0", "0.00", "0.000"])
def test_zero_cpu_request_is_rejected(cpu: str) -> None:
    body = _sample()
    body.resources.cpus_per_node = cpu
    with pytest.raises(ApiError) as raised:
        submission._validate_request(body)
    assert raised.value.status_code == 422
    assert raised.value.code == "validation_error"


def test_positive_fractional_cpu_request_is_valid() -> None:
    body = _sample()
    body.resources.cpus_per_node = "0.01"
    resources, _parameters = submission._validate_request(body)
    assert resources.cpus_per_node == "0.01"


def test_configured_image_is_embedded_in_rayjob_head_and_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "quay.io/example/ray:2.58.0"
    _mock_sdk(monkeypatch)
    submission.CodeFlareJobBackend(image)._submit_sync(
        "team-alpha", _sample(), CallerIdentity("alice", None, ()), "caller-token"
    )
    cluster = FakeRayJob.instances[0].kwargs["cluster_config"]
    assert cluster.image == image
    spec = build_ray_cluster_spec(cluster, "pvc-training-cluster")
    assert spec["rayVersion"] == "2.58.0"
    assert spec["headGroupSpec"]["template"].spec.containers[0].image == image
    assert spec["workerGroupSpecs"][0]["template"].spec.containers[0].image == image


@pytest.mark.parametrize("num_nodes", [1, 2])
def test_sdk_submission_metadata_and_topology_are_readable(
    monkeypatch: pytest.MonkeyPatch, num_nodes: int
) -> None:
    core = _mock_sdk(monkeypatch)
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["resources"]["num_nodes"] = num_nodes
    if num_nodes > 1:
        payload["algorithm"] = {"name": "lora_grpo"}
    authorizer = FakeAuthorizer()
    http = TestClient(
        create_app(
            FakeAuthenticator(),
            authorizer,
            submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    )
    headers = {"Authorization": "Bearer caller-token"}
    base = "/api/v1/projects/team-alpha/training-jobs"
    response = http.post(base, json=payload, headers=headers)
    assert response.status_code == 202
    assert response.headers["location"] == base + "/" + payload["name"]

    kwargs = FakeRayJob.instances[0].kwargs
    with client.ApiClient() as serializer:
        cluster = serializer.sanitize_for_serialization(
            build_ray_cluster_spec(kwargs["cluster_config"], "pvc-training-cluster")
        )
    record = {
        "metadata": {
            "name": payload["name"],
            "creationTimestamp": "2026-10-27T09:00:00Z",
            "annotations": kwargs["annotations"],
            "labels": {"kueue.x-k8s.io/queue-name": payload["queue_name"]},
        },
        "spec": {"rayClusterSpec": cluster},
    }

    def custom_api(api: FakeApiClient) -> SimpleNamespace:
        assert api.token == "caller-token"
        return SimpleNamespace(
            list_namespaced_custom_object=lambda *args, **kwargs: {"items": [record]},
            get_namespaced_custom_object=lambda *args, **kwargs: record,
        )

    monkeypatch.setattr(jobs, "caller_client", lambda request: FakeApiClient("caller-token"))
    monkeypatch.setattr(jobs.client, "CustomObjectsApi", custom_api)
    monkeypatch.setattr(core, "read_namespace", lambda *args, **kwargs: object(), raising=False)
    listing = http.get(base, headers=headers)
    assert listing.status_code == 200
    assert listing.json() == {
        "items": [
            {
                "id": payload["name"],
                "status": "PENDING",
                "created_at": "2026-10-27T09:00:00Z",
                "queue_name": payload["queue_name"],
            }
        ],
        "total_count": 1,
    }
    response = http.get(response.headers["location"], headers=headers)
    assert response.status_code == 200
    assert response.json()["submission"] == payload
    assert response.json()["submitted_by"] == "alice"
    if num_nodes == 1:
        assert response.json()["resources"] == payload["resources"]
    else:
        # A GPU-free head and GPU workers are heterogeneous; PR #8 omits resources.
        assert "resources" not in response.json()
    assert authorizer.calls == [
        ProjectPermission("ray.io", "rayjobs", "create"),
        ProjectPermission("ray.io", "rayjobs", "list"),
        ProjectPermission("ray.io", "rayjobs", "get", payload["name"]),
    ]


def test_omitted_basic_options_use_contract_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_sdk(monkeypatch)
    entry_config: dict[str, Any] = {}

    def record_entrypoint(configuration: dict[str, Any]) -> str:
        entry_config.update(configuration)
        return "python -c 'pass'"

    monkeypatch.setattr(submission, "_entrypoint", record_entrypoint)
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    for key in ("queue_name", "resources", "output"):
        payload.pop(key)
    payload["algorithm"].pop("parameters")
    body = CreateJobRequest.model_validate(payload)

    submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        "team-alpha", body, CallerIdentity("alice", None, ()), "caller-token"
    )

    kwargs = FakeRayJob.instances[0].kwargs
    cluster = kwargs["cluster_config"]
    assert kwargs["local_queue"] is None
    assert cluster.num_workers == 0
    assert cluster.head_cpu_requests == "4"
    assert cluster.head_memory_requests == "16Gi"
    assert cluster.head_extended_resource_requests == {"nvidia.com/gpu": 1}
    assert len(cluster.volumes) == 2
    assert entry_config["parameters"] == {}
    assert "output" not in entry_config


@pytest.mark.parametrize(
    "example,expected_reads",
    [
        (
            "huggingface_inputs_s3_output",
            ["hf-secret-token", "checkpoint-storage"],
        ),
        ("s3_inputs", ["model-storage", "training-data", "checkpoint-storage"]),
    ],
)
def test_secret_backed_locators_use_caller_client_and_mount_references_only(
    monkeypatch: pytest.MonkeyPatch, example: str, expected_reads: list[str]
) -> None:
    core = _mock_sdk(monkeypatch)
    body = _sample(example)
    submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        "team-alpha", body, CallerIdentity("alice", None, ()), "caller-token"
    )
    assert [name for name, _ in core.secret_reads] == expected_reads
    cluster = FakeRayJob.instances[0].kwargs["cluster_config"]
    assert all(volume.secret is not None for volume in cluster.volumes)
    assert "opaque-value" not in FakeRayJob.instances[0].kwargs["entrypoint"]
    assert "opaque-value" not in str(FakeRayJob.instances[0].kwargs["annotations"])


@pytest.mark.anyio
async def test_sdk_conflict_maps_to_documented_409(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_sdk(monkeypatch)
    FakeRayJob.error = ApiException(status=409, reason="Already Exists")
    submitter = submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")
    with pytest.raises(ApiError) as raised:
        await submitter.submit(
            "team-alpha", _sample(), CallerIdentity("alice", None, ()), "caller-token"
        )
    assert raised.value.status_code == 409
    assert raised.value.code == "already_exists"
    assert raised.value.message == (
        "Training job 'pvc-training' already exists in project 'team-alpha'."
    )


@pytest.mark.parametrize(
    "example,kind,name,method",
    [
        ("pvc_inputs_output", "PersistentVolumeClaim", "model-storage", "pvc"),
        ("pvc_inputs_output", "PersistentVolumeClaim", "training-data", "pvc"),
        ("pvc_inputs_output", "PersistentVolumeClaim", "training-output", "pvc"),
        ("huggingface_inputs_s3_output", "Secret", "hf-secret-token", "secret"),
        ("s3_inputs", "Secret", "model-storage", "secret"),
    ],
)
def test_missing_submission_resources_identify_reference_and_project(
    monkeypatch, example, kind, name, method
):
    core = _mock_sdk(monkeypatch)
    attribute = (
        "read_namespaced_persistent_volume_claim" if method == "pvc" else "read_namespaced_secret"
    )
    original = getattr(core, attribute)

    def read(resource_name, namespace, **kwargs):
        if resource_name == name:
            error = ApiException(status=404, reason="sensitive-backend-body")
            error.body = "caller-token"
            raise error
        return original(resource_name, namespace, **kwargs)

    monkeypatch.setattr(core, attribute, read)
    http = TestClient(
        create_app(
            FakeAuthenticator(),
            FakeAuthorizer(),
            submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    )
    response = http.post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=_sample(example).model_dump(mode="json", exclude_unset=True),
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 404
    assert response.json() == {
        "code": "not_found",
        "message": f"{kind} '{name}' was not found in project 'team-alpha'.",
    }
    assert "sensitive-backend-body" not in response.text
    assert "caller-token" not in response.text
    assert FakeRayJob.instances == []


def test_submission_does_not_mutate_process_global_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    previous_client = FakeApiClient("previous-caller")
    monkeypatch.setattr(codeflare_auth, "api_client", previous_client)
    submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        "team-alpha", _sample(), CallerIdentity("alice", None, ()), "caller-token"
    )
    assert FakeRayJob.instances[0].client_at_submit.token == "caller-token"
    assert codeflare_auth.api_client is previous_client


def test_parallel_submissions_keep_each_callers_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    barrier = Barrier(2)
    previous_client = FakeApiClient("previous-caller")
    monkeypatch.setattr(codeflare_auth, "api_client", previous_client)

    def controlled_submit(job: FakeRayJob) -> str:
        job.client_at_submit = job.kwargs["api_client"]
        barrier.wait(timeout=5)
        assert codeflare_auth.api_client is previous_client
        return str(job.kwargs["job_name"])

    monkeypatch.setattr(FakeRayJob, "submit", controlled_submit)
    submitters = [submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0") for _ in range(2)]

    def submit(index: int, name: str, token: str) -> str:
        body = _sample()
        body.name = name
        return submitters[index]._submit_sync(
            "team-alpha", body, CallerIdentity(name, None, ()), token
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(submit, 0, "job-a", "caller-a")
        second = pool.submit(submit, 1, "job-b", "caller-b")
        assert first.result(timeout=5) == "job-a"
        assert second.result(timeout=5) == "job-b"

    assert {job.client_at_submit.token for job in FakeRayJob.instances} == {"caller-a", "caller-b"}
    assert codeflare_auth.api_client is previous_client


@pytest.mark.parametrize(
    "backend_status,api_status",
    [(401, 401), (403, 403), (404, 404), (409, 409), (422, 422), (500, 503)],
)
def test_sdk_owned_create_preserves_kubernetes_failure_status(
    monkeypatch: pytest.MonkeyPatch, backend_status: int, api_status: int
) -> None:
    observed: dict[str, Any] = {}

    def reject_create(
        _self: client.ApiClient, resource_path: str, method: str, *args: Any, **kwargs: Any
    ) -> None:
        observed.update(path=resource_path, method=method, timeout=kwargs.get("_request_timeout"))
        raise ApiException(status=backend_status, reason="SDK create failed")

    monkeypatch.setattr(client.ApiClient, "call_api", reject_create)
    api_client = submission._CallerApiClient(client.Configuration())
    sdk_transport = object.__new__(RayjobApi)
    sdk_transport.api = client.CustomObjectsApi(api_client)
    sdk_transport.kube_config = None
    with pytest.raises(submission._SdkSubmissionError) as raised:
        sdk_transport.submit_job("team-alpha", {"metadata": {"name": "job-1"}})
    assert observed == {
        "path": "/apis/{group}/{version}/namespaces/{namespace}/{plural}",
        "method": "POST",
        "timeout": 10,
    }
    assert raised.value.error.status == backend_status
    assert (
        submission._map_backend_error(raised.value, "team-alpha", "job-1").status_code == api_status
    )

    with pytest.raises(ApiException):
        api_client.call_api("/api/v1/namespaces/{namespace}/pods", "GET")


def test_caller_client_times_out_all_sdk_calls_without_overriding_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[int] = []

    def reject_call(
        _self: client.ApiClient, _resource_path: str, _method: str, *args: Any, **kwargs: Any
    ) -> None:
        timeouts.append(kwargs["_request_timeout"])
        raise ApiException(status=403)

    monkeypatch.setattr(client.ApiClient, "call_api", reject_call)
    api_client = submission._CallerApiClient(client.Configuration())

    with pytest.raises(ApiException):
        api_client.call_api("/api/v1/namespaces/{namespace}/pods", "GET")
    with pytest.raises(ApiException):
        api_client.call_api("/api/v1/namespaces/{namespace}/pods", "GET", _request_timeout=None)
    with pytest.raises(ApiException):
        api_client.call_api("/api/v1/namespaces/{namespace}/pods", "GET", _request_timeout=3)
    with pytest.raises(submission._SdkSubmissionError):
        api_client.call_api(
            submission._RAYJOB_CREATE_PATH,
            "POST",
            {"group": "ray.io", "version": "v1", "plural": "rayjobs"},
            _request_timeout=3,
        )
    assert timeouts == [submission._K8S_TIMEOUT_SECONDS] * 2 + [3, 3]


def test_caller_client_never_falls_back_to_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_incluster_config(**kwargs: Any) -> None:
        pytest.fail("Caller clients must not load service-account credentials.")

    default = client.Configuration()
    default.api_key["authorization"] = "Bearer service-account-token"
    default.refresh_api_key_hook = lambda configuration: pytest.fail("Refreshed pod credentials.")
    monkeypatch.setattr(client.Configuration, "_default", default)
    monkeypatch.setattr(config, "load_incluster_config", forbidden_incluster_config)
    with submission._caller_api_client("caller-token") as api:
        assert api.configuration.api_key == {"authorization": "Bearer caller-token"}
        assert api.configuration.refresh_api_key_hook is None
        assert api.configuration.get_api_key_with_prefix("authorization") == "Bearer caller-token"
    assert default.api_key["authorization"] == "Bearer service-account-token"
    assert client.Configuration._default is default


@pytest.mark.parametrize("overrides", [False, True])
def test_submission_and_operations_share_caller_configuration(monkeypatch, overrides):
    def fake_incluster_config(*, client_configuration, try_refresh_token):
        client_configuration.host = "https://incluster.example:6443"
        client_configuration.ssl_ca_cert = "/mounted/service-account/ca.crt"

    monkeypatch.setattr(config, "load_incluster_config", fake_incluster_config)
    expected_host = "https://kubernetes.default.svc"
    expected_ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    for name in ("TRAINING_SERVICE_KUBERNETES_API_URL", "TRAINING_SERVICE_KUBERNETES_CA_FILE"):
        monkeypatch.delenv(name, raising=False)
    if overrides:
        expected_host = "https://override.example:6443"
        expected_ca = "/custom/cluster-ca.crt"
        monkeypatch.setenv("TRAINING_SERVICE_KUBERNETES_API_URL", expected_host)
        monkeypatch.setenv("TRAINING_SERVICE_KUBERNETES_CA_FILE", expected_ca)
    default = client.Configuration._default
    request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", b"Bearer caller-token")],
            "state": {"caller_identity": CallerIdentity("alice", "uid-a", ())},
        }
    )
    with (
        submission._caller_api_client("caller-token") as submit_api,
        kubernetes_client.caller_client(request) as read_api,
    ):
        assert submit_api.configuration is not read_api.configuration
        for configuration in (submit_api.configuration, read_api.configuration):
            assert configuration.host == expected_host
            assert configuration.ssl_ca_cert == expected_ca
            assert configuration.verify_ssl is True
            assert configuration.retries == 0
            assert configuration.api_key == {"authorization": "Bearer caller-token"}
            assert "authorization" not in configuration.api_key_prefix
            assert configuration.refresh_api_key_hook is None
    assert client.Configuration._default is default


@pytest.mark.parametrize("operation", ["submission", "read"])
@pytest.mark.parametrize(
    "host",
    [
        "http://cluster.example:6443",
        "",
        "cluster.example:6443",
        "https://",
        "https://user:secret@cluster.example",
        "https://cluster.example?token=secret",
        "https://cluster.example/#secret",
        "https://cluster.example:invalid",
    ],
)
def test_invalid_kubernetes_url_never_creates_caller_client(monkeypatch, operation, host):
    def forbidden_client(configuration):
        pytest.fail("Invalid Kubernetes URLs must be rejected before creating a client.")

    monkeypatch.setenv("TRAINING_SERVICE_KUBERNETES_API_URL", host)
    monkeypatch.setattr(submission, "_CallerApiClient", forbidden_client)
    monkeypatch.setattr(kubernetes_client.client, "ApiClient", forbidden_client)
    with pytest.raises(ApiError) as raised:
        if operation == "submission":
            submission._caller_api_client("caller-token")
        else:
            request = Request(
                {
                    "type": "http",
                    "headers": [(b"authorization", b"Bearer caller-token")],
                    "state": {"caller_identity": CallerIdentity("alice", "uid-a", ())},
                }
            )
            with kubernetes_client.caller_client(request):
                pytest.fail("Invalid Kubernetes URL was accepted.")
    assert raised.value.status_code == 503
    assert raised.value.code == "backend_unavailable"
    assert "HTTPS" in raised.value.message
    assert "caller-token" not in raised.value.message
    assert "secret" not in raised.value.message


@pytest.mark.parametrize(
    "backend_status,api_status",
    [(401, 401), (403, 403), (404, 404), (409, 409), (422, 422), (500, 503)],
)
def test_kubernetes_errors_map_to_documented_statuses(backend_status: int, api_status: int) -> None:
    error = submission._map_backend_error(
        ApiException(status=backend_status), "team-alpha", "job-1"
    )
    assert error.status_code == api_status
    assert error.message
    ErrorResponse.model_validate({"code": error.code, "message": error.message})


def test_unsupported_backend_and_unsafe_pvc_path_fail_before_sdk() -> None:
    body = _sample()
    body.backend = "kft"
    with pytest.raises(ApiError) as raised:
        submission._validate_request(body)
    assert raised.value.status_code == 501
    assert raised.value.code == "not_implemented"
    with pytest.raises(ApiError) as raised:
        submission._parse_location("pvc://claim/safe/%2e%2e/secret")
    assert raised.value.status_code == 422
    with pytest.raises(ApiError) as raised:
        submission._parse_location("pvc://claim/file%00name")
    assert raised.value.status_code == 422


@pytest.mark.parametrize("name", ["sft", "osft", "lora_sft", "lora_grpo"])
def test_discovered_algorithm_is_forwarded_without_hardcoded_parameters(
    monkeypatch, runtime_catalog, name
):
    _mock_sdk(monkeypatch)
    metadata = json.loads(runtime_catalog.read_text())
    algorithm = next(item for item in metadata["algorithms"] if item["name"] == name)
    algorithm["parameter_schema"]["properties"]["runtime_added_parameter"] = {"type": "integer"}
    runtime_catalog.write_text(json.dumps(metadata))
    body = _sample()
    body.algorithm.name = name
    body.algorithm.parameters = {
        **(_OSFT_EXAMPLE_PARAMETERS if name == "osft" else {}),
        "runtime_added_parameter": 0,
    }
    if name == "lora_grpo":
        body.resources.num_nodes = 2
    entrypoint = {}

    def record(configuration):
        entrypoint.update(configuration)
        return "python -c 'pass'"

    monkeypatch.setattr(jobs, "_entrypoint", record)
    jobs.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        "team-alpha", body, CallerIdentity("alice", None, ()), "token"
    )
    assert entrypoint["algorithm"] == name
    assert entrypoint["parameters"] == body.algorithm.parameters


@pytest.mark.parametrize("failure", ["algorithm", "profile", "parameter", "type", "catalog"])
def test_api_rejects_runtime_incompatibility_before_kubernetes(
    monkeypatch, runtime_catalog, failure
):
    from unittest.mock import Mock

    body = _sample().model_dump(mode="json", exclude_unset=True)
    if failure == "algorithm":
        metadata = json.loads(runtime_catalog.read_text())
        metadata["algorithms"] = [a for a in metadata["algorithms"] if a["name"] != "sft"]
        runtime_catalog.write_text(json.dumps(metadata))
        body["algorithm"]["name"] = "sft"
        context = "not available"
    elif failure == "profile":
        body["algorithm"]["name"] = "grpo"
        body["algorithm"]["parameters"] = {}
        context = "no execution profile"
    elif failure == "parameter":
        body["algorithm"]["parameters"]["unsupported_in_old_image"] = 1
        context = "unsupported_in_old_image"
    elif failure == "type":
        body["algorithm"]["parameters"]["learning_rate"] = "bad-value"
        context = "learning_rate"
    else:
        runtime_catalog.unlink()
        context = "discover-algorithms"
    factory = Mock(side_effect=AssertionError("Kubernetes must not be called"))
    monkeypatch.setattr(jobs, "_caller_api_client", factory)
    http = TestClient(
        create_app(
            FakeAuthenticator(),
            FakeAuthorizer(),
            backend=jobs.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    )
    response = http.post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=body,
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == (503 if failure == "catalog" else 422)
    assert context in response.json()["message"]
    assert response.json()["code"] == (
        "backend_unavailable" if failure == "catalog" else "validation_error"
    )
    factory.assert_not_called()


def test_metadata_is_added_to_sdk_built_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RayJob,
        "_build_rayjob_cr",
        lambda _self: {"metadata": {"name": "test-job"}, "spec": {"entrypoint": "python -c"}},
    )
    job = object.__new__(submission._MetadataRayJob)
    job._submission_annotations = {"training.rhoai.io/submitted-by": "alice"}
    resource = job._build_rayjob_cr()
    assert resource["metadata"] == {
        "name": "test-job",
        "annotations": {"training.rhoai.io/submitted-by": "alice"},
    }
    assert resource["spec"] == {"entrypoint": "python -c"}


def test_alpaca_jsonl_uses_training_hub_alpaca_format(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"instruction":"question","input":"","output":"answer"}\n', encoding="utf-8")
    assert _dataset_format(str(path)) == "alpaca"
    path.write_text('{"messages":[]}\n', encoding="utf-8")
    assert _dataset_format(str(path)) is None


_EXPECTED_TH_BACKENDS = {
    "sft": "instructlab-training",
    "osft": "mini-trainer",
    "lora_sft": "unsloth",
    "lora_grpo": "verl",
}


_OSFT_EXAMPLE_PARAMETERS = {
    "unfreeze_rank_ratio": 0.1,
    "max_seq_len": 512,
    "max_tokens_per_gpu": 512,
    "effective_batch_size": 4,
    "learning_rate": 1e-5,
}


def test_all_post_contract_examples_match_algorithm_validation() -> None:
    examples = load_contract()["paths"]["/projects/{project}/training-jobs"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]
    for example in examples.values():
        submission._validate_request(CreateJobRequest.model_validate(example["value"]))


def test_runtime_catalog_covers_registered_execution_profiles() -> None:
    names = {item["name"] for item in catalog.load_catalog()["algorithms"]}
    assert set(ray_entrypoint._BACKENDS) <= names
    algorithm_spec = load_contract()["components"]["schemas"]["AlgorithmSpec"]
    assert "x-parameter-schemas" not in algorithm_spec
    assert "properties" not in algorithm_spec["properties"]["parameters"]
    for name in ray_entrypoint._BACKENDS:
        assert get_backend(name).name == name
    with pytest.raises(ValueError, match="Unsupported Training Hub algorithm"):
        get_backend("unreviewed_algorithm")


@pytest.mark.parametrize("algorithm", ["sft", "osft", "lora_sft"])
def test_head_local_algorithms_reject_unused_ray_workers_before_sdk(
    monkeypatch: pytest.MonkeyPatch, algorithm: str
) -> None:
    core = _mock_sdk(monkeypatch)
    body = _sample()
    body.algorithm.name = algorithm
    body.algorithm.parameters = _OSFT_EXAMPLE_PARAMETERS if algorithm == "osft" else {}
    body.resources.num_nodes = 2
    with pytest.raises(ApiError, match="num_nodes=1") as raised:
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
            "team-alpha", body, CallerIdentity("alice", None, ()), "caller-token"
        )
    assert raised.value.status_code == 422
    assert core.pvc_reads == []
    assert FakeRayJob.instances == []


def test_lora_multi_gpu_requires_explicit_model_splitting() -> None:
    body = _sample()
    body.resources.gpus_per_node = 2
    with pytest.raises(ApiError, match="enable_model_splitting"):
        submission._validate_request(body)
    body.algorithm.parameters["enable_model_splitting"] = True
    resources, parameters = submission._validate_request(body)
    assert resources.gpus_per_node == 2
    assert parameters["enable_model_splitting"] is True


def test_conflicting_lora_precision_requires_explicit_bf16_false() -> None:
    assert validate_algorithm_parameters("lora_sft", {"fp16": True, "bf16": False}) == {
        "fp16": True,
        "bf16": False,
    }


@pytest.mark.parametrize("algorithm", ["sft", "osft", "lora_sft", "lora_grpo"])
@pytest.mark.parametrize(
    "parameter",
    ["nproc_per_node", "nnodes", "node_rank", "backend", "callbacks", "checkpoint_storage"],
)
def test_algorithm_parameters_cannot_override_platform_or_python_configuration(
    algorithm: str, parameter: str
) -> None:
    values = dict(_OSFT_EXAMPLE_PARAMETERS) if algorithm == "osft" else {}
    values[parameter] = "caller-controlled"
    with pytest.raises(ApiError) as raised:
        validate_algorithm_parameters(algorithm, values)
    assert raised.value.status_code == 422


def test_real_sdk_submits_with_caller_client_and_configured_ray_image(
    monkeypatch: pytest.MonkeyPatch,
    runtime_catalog,
) -> None:
    # CodeFlare's vendored wrappers load a default kubeconfig in __init__ even
    # though RayJob immediately replaces their clients with the caller's client.
    # Mock only that unused constructor side effect; keep the real SDK submit
    # and RayJob manifest-building paths under test, without a cluster login.
    def skip_default_config(self: RayjobApi | RayClusterApi) -> None:
        self.kube_config = None

    monkeypatch.setattr(RayjobApi, "__init__", skip_default_config)
    monkeypatch.setattr(RayClusterApi, "__init__", skip_default_config)
    configuration = client.Configuration()
    configuration.api_key["authorization"] = "Bearer caller-token"
    caller_client = submission._CallerApiClient(configuration)
    previous_client = client.ApiClient(client.Configuration())
    monkeypatch.setattr(codeflare_auth, "api_client", previous_client)
    monkeypatch.setattr(submission, "_caller_api_client", lambda _token: caller_client)
    monkeypatch.setattr(submission.client, "CoreV1Api", lambda _api_client=None: FakeCoreApi())
    created: list[dict[str, Any]] = []

    def record_create(api: client.CustomObjectsApi, **kwargs: Any) -> dict[str, Any]:
        assert api.api_client is caller_client
        assert kwargs["namespace"] == "team-alpha"
        created.append(kwargs["body"])
        return kwargs["body"]

    monkeypatch.setattr(client.CustomObjectsApi, "create_namespaced_custom_object", record_create)
    image = "quay.io/modh/ray:2.58.0-py312-cu129-th081"
    metadata = json.loads(runtime_catalog.read_text())
    metadata["runtime"]["image"] = image
    runtime_catalog.write_text(json.dumps(metadata))
    identifier = submission.CodeFlareJobBackend(image)._submit_sync(
        "team-alpha", _sample(), CallerIdentity("alice", None, ()), "caller-token"
    )

    assert identifier == "pvc-training"
    assert len(created) == 1
    assert created[0]["metadata"]["annotations"]["training.rhoai.io/submitted-by"] == "alice"
    assert created[0]["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == (
        "team-alpha-local-queue"
    )
    spec = created[0]["spec"]["rayClusterSpec"]
    assert spec["rayVersion"] == "2.58.0"
    assert spec["headGroupSpec"]["template"].spec.containers[0].image == image
    assert spec["workerGroupSpecs"][0]["template"].spec.containers[0].image == image
    assert codeflare_auth.api_client is previous_client

    with pytest.raises(ApiError, match="does not match the runtime image") as raised:
        submission.CodeFlareJobBackend("quay.io/modh/ray:2.55.1-py312-cu129-th081")._submit_sync(
            "team-alpha", _sample(), CallerIdentity("alice", None, ()), "caller-token"
        )
    assert (
        submission._map_backend_error(raised.value, "team-alpha", "pvc-training").status_code == 503
    )
    assert len(created) == 1


@pytest.mark.parametrize(
    "algorithm,parameters",
    [
        (
            "sft",
            {
                "max_seq_len": 512,
                "max_tokens_per_gpu": 512,
                "effective_batch_size": 4,
                "learning_rate": 1e-5,
                "num_epochs": 1,
            },
        ),
        (
            "osft",
            {**_OSFT_EXAMPLE_PARAMETERS, "lr_scheduler_kwargs": {"num_cycles": "1"}},
        ),
        (
            "lora_sft",
            {"lora_r": 16, "lora_alpha": 32, "num_epochs": 1, "max_grad_norm": 0.3},
        ),
        (
            "lora_grpo",
            {
                "num_iterations": 5,
                "group_size": 4,
                "n_train": 200,
                "n_val": 10,
                "lora_r": 16,
                "lora_alpha": 8,
                "learning_rate": 1e-5,
                "gpu_memory_utilization": 0.2,
            },
        ),
    ],
)
def test_example_algorithm_parameters_are_preserved(
    algorithm: str, parameters: dict[str, Any]
) -> None:
    assert validate_algorithm_parameters(algorithm, parameters) == parameters


@pytest.mark.parametrize(
    "algorithm,parameters",
    [
        ("other", {}),
        ("sft", {"max_seq_len": True}),
        ("sft", {"learning_rate": float("nan")}),
        ("sft", {"is_pretraining": True}),
        ("sft", {"beta1": 0.9}),
        ("sft", {"block_size": 512}),
        ("osft", {"unfreeze_rank_ratio": 0.1}),
        ("osft", {**_OSFT_EXAMPLE_PARAMETERS, "lr_scheduler_kwargs": {"num_cycles": 1}}),
        (
            "osft",
            {
                **_OSFT_EXAMPLE_PARAMETERS,
                "is_pretraining": True,
                "block_size": 512,
                "unmask_messages": True,
            },
        ),
        ("lora_sft", {"lora_r": "16"}),
        ("lora_sft", {"lora_r": 16, "lora_rank": 16}),
        ("lora_sft", {"load_in_4bit": True, "load_in_8bit": True}),
        ("lora_sft", {"assistant_only_loss": False}),
        ("lora_sft", {"fp16": True}),
        ("lora_sft", {"seed": "1"}),
        ("lora_sft", {"use_rslora": "true"}),
        ("lora_grpo", {"tasks_per_iteration": 10}),
        ("lora_grpo", {"n_gpus": 4}),
        ("lora_grpo", {"param_offload": "true"}),
        ("osft", {**_OSFT_EXAMPLE_PARAMETERS, "eps": 10**400}),
    ],
)
def test_invalid_algorithm_parameters_get_422(algorithm: str, parameters: dict[str, Any]) -> None:
    with pytest.raises(ApiError) as raised:
        validate_algorithm_parameters(algorithm, parameters)
    assert raised.value.status_code == 422
    assert raised.value.code == "validation_error"


def test_osft_integral_json_numbers_are_normalized_for_training_hub_float_checks() -> None:
    parameters = {**_OSFT_EXAMPLE_PARAMETERS, "unfreeze_rank_ratio": 1}
    mapped = validate_algorithm_parameters("osft", parameters)
    assert mapped["unfreeze_rank_ratio"] == 1.0
    assert type(mapped["unfreeze_rank_ratio"]) is float
    assert parameters["unfreeze_rank_ratio"] == 1


@pytest.mark.parametrize(
    "algorithm, parameters",
    [
        ("lora_sft", {"flash_attention": True, "eval_steps": 5, "tf32": True}),
        ("lora_grpo", {"max_lora_rank": 8, "temperature": 0.7, "concurrency": 4}),
        ("lora_grpo", {"target_modules": ["q_proj"], "max_grad_norm": 0.1}),
    ],
)
def test_runtime_parameters_are_not_restricted_to_the_old_common_subset(algorithm, parameters):
    assert validate_algorithm_parameters(algorithm, parameters) == parameters


def test_advertised_backend_must_match_the_execution_profile():
    assert validate_algorithm_parameters("lora_grpo", {"backend": "verl"}) == {}
    with pytest.raises(ApiError, match="requires Training Hub backend 'verl'"):
        validate_algorithm_parameters("lora_grpo", {"backend": "art"})


def test_discovery_and_http_submission_share_runtime_parameter_changes(
    monkeypatch, runtime_catalog
):
    _mock_sdk(monkeypatch)
    metadata = json.loads(runtime_catalog.read_text())
    descriptor = next(a for a in metadata["algorithms"] if a["name"] == "lora_sft")
    descriptor["parameter_schema"]["properties"]["future_runtime_option"] = {"type": "integer"}
    runtime_catalog.write_text(json.dumps(metadata))
    http = TestClient(
        create_app(
            FakeAuthenticator(),
            FakeAuthorizer(),
            backend=jobs.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    )
    headers = {"Authorization": "Bearer caller-token"}
    details = "/api/v1/algorithms/lora_sft"
    assert "future_runtime_option" in http.get(details, headers=headers).json()["parameters"]
    body = _sample().model_dump(mode="json", exclude_unset=True)
    body["algorithm"]["parameters"]["future_runtime_option"] = 1
    assert (
        http.post(
            "/api/v1/projects/team-alpha/training-jobs", json=body, headers=headers
        ).status_code
        == 202
    )
    descriptor["parameter_schema"]["properties"].pop("future_runtime_option")
    runtime_catalog.write_text(json.dumps(metadata))
    catalog._load_catalog.cache_clear()  # Simulate a rollout onto the older runtime.
    assert "future_runtime_option" not in http.get(details, headers=headers).json()["parameters"]
    lookup = Mock(side_effect=AssertionError("Kubernetes must not be called"))
    monkeypatch.setattr(jobs, "_caller_api_client", lookup)
    response = http.post("/api/v1/projects/team-alpha/training-jobs", json=body, headers=headers)
    assert response.status_code == 422
    assert "future_runtime_option" in response.json()["message"]
    lookup.assert_not_called()


@pytest.mark.parametrize("algorithm", ["sft", "osft", "lora_sft", "lora_grpo"])
def test_every_exposed_algorithm_override_reaches_training_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, algorithm: str
) -> None:
    descriptor = next(a for a in catalog.load_catalog()["algorithms"] if a["name"] == algorithm)
    properties = descriptor["parameter_schema"]["properties"]
    values: dict[str, Any] = {}
    for key, spec in properties.items():
        if key == "backend":
            continue
        if "const" in spec:
            values[key] = spec["const"]
            continue
        if "anyOf" in spec:
            spec = next(option for option in spec["anyOf"] if option.get("type") != "null")
        if "enum" in spec:
            values[key] = spec["enum"][0]
        elif spec["type"] == "boolean":
            values[key] = key == "is_pretraining"
        elif spec["type"] == "integer":
            values[key] = max(spec.get("minimum", 0), 1)
        elif spec["type"] == "number":
            values[key] = 0.5
        elif spec["type"] == "array":
            values[key] = ["q_proj"]
        elif spec["type"] == "object":
            item = 1 if spec.get("additionalProperties", {}).get("type") == "integer" else "1"
            values[key] = {"num_cycles": item}
        else:
            values[key] = "value"

    mapped = validate_algorithm_parameters(algorithm, values)
    calls: list[dict[str, Any]] = []

    def factory(name: str, backend: str) -> Any:
        assert name == algorithm
        assert backend == _EXPECTED_TH_BACKENDS[algorithm]
        return SimpleNamespace(train=lambda **kw: calls.append(kw))

    monkeypatch.setitem(sys.modules, "training_hub", SimpleNamespace(create_algorithm=factory))
    ray_entrypoint._run_training(algorithm, "/model", "/data.jsonl", tmp_path, mapped, 4, 1)

    assert len(calls) == 1
    for key, value in mapped.items():
        assert calls[0][key] == value


@pytest.mark.parametrize(
    "algorithm,parameters,gpus,nodes",
    [
        ("sft", {"max_tokens_per_gpu": 512}, 1, 1),
        ("osft", _OSFT_EXAMPLE_PARAMETERS, 1, 1),
        ("lora_sft", {"lora_r": 16, "lora_alpha": 32}, 1, 1),
        ("lora_grpo", {"group_size": 4, "lora_r": 16}, 4, 2),
    ],
)
def test_sdk_submission_dispatches_example_algorithm_and_gpu_topology(
    monkeypatch: pytest.MonkeyPatch,
    algorithm: str,
    parameters: dict[str, Any],
    gpus: int,
    nodes: int,
) -> None:
    core = _mock_sdk(monkeypatch)
    configuration: dict[str, Any] = {}
    monkeypatch.setattr(
        submission,
        "_entrypoint",
        lambda value: configuration.update(value) or "python -c 'pass'",
    )
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["algorithm"] = {"name": algorithm, "parameters": parameters}
    payload["resources"]["num_nodes"] = nodes
    payload["resources"]["gpus_per_node"] = gpus
    if algorithm == "lora_grpo":
        payload["model"]["uri"] = "hf://hf-secret-token/Qwen/Qwen3-4B"
        payload["dataset"]["uri"] = "hf://hf-secret-token/Agent-Ark/Toucan-1.5M"
    body = CreateJobRequest.model_validate(payload)

    identifier = submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        "team-alpha", body, CallerIdentity("alice", None, ()), "caller-token"
    )

    assert identifier == "pvc-training"
    assert configuration["algorithm"] == algorithm
    assert configuration["parameters"] == parameters
    assert configuration["gpus_per_node"] == gpus
    assert configuration["num_workers"] == nodes - 1
    cluster = FakeRayJob.instances[0].kwargs["cluster_config"]
    assert cluster.head_extended_resource_requests == (
        {} if algorithm == "lora_grpo" else {"nvidia.com/gpu": gpus}
    )
    assert cluster.worker_extended_resource_requests == {"nvidia.com/gpu": gpus}
    assert cluster.num_workers == nodes - 1
    if algorithm == "lora_grpo":
        assert core.secret_reads == [("hf-secret-token", "team-alpha")]
        assert cluster.envs["HF_TOKEN_PATH"] == "/mnt/training-api/credentials/0/HF_TOKEN"
        assert cluster.head_cpu_requests == "4"
        assert cluster.head_memory_requests == "32Gi"
        assert cluster.worker_cpu_requests == "4"
        assert cluster.worker_memory_requests == "16Gi"
        built = build_ray_cluster_spec(cluster, "grpo-toucan-cluster")
        head = built["headGroupSpec"]["template"].spec.containers[0]
        worker = built["workerGroupSpecs"][0]["template"].spec.containers[0]
        assert "nvidia.com/gpu" not in head.resources.requests
        assert worker.resources.requests["nvidia.com/gpu"] == 4
        assert any(env.name == "HF_TOKEN_PATH" for env in head.env)
        assert any(env.name == "HF_TOKEN_PATH" for env in worker.env)
        assert any(
            mount.mount_path == "/mnt/training-api/credentials/0" for mount in head.volume_mounts
        )
        assert any(
            mount.mount_path == "/mnt/training-api/credentials/0" for mount in worker.volume_mounts
        )
        assert "opaque-value" not in str(FakeRayJob.instances[0].kwargs)
    else:
        assert "HF_TOKEN_PATH" not in cluster.envs


@pytest.mark.parametrize(
    "algorithm,parameters",
    [
        ("sft", {"max_tokens_per_gpu": 512}),
        ("osft", _OSFT_EXAMPLE_PARAMETERS),
        ("lora_sft", {"lora_r": 16}),
        ("lora_grpo", {"group_size": 4, "lora_r": 16}),
    ],
)
def test_http_post_accepts_adapted_ray_example_parameters(
    monkeypatch: pytest.MonkeyPatch, algorithm: str, parameters: dict[str, Any]
) -> None:
    _mock_sdk(monkeypatch)
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["algorithm"] = {"name": algorithm, "parameters": parameters}
    if algorithm == "lora_grpo":
        payload["model"]["uri"] = "hf://hf-secret-token/Qwen/Qwen3-4B"
        payload["dataset"]["uri"] = "hf://hf-secret-token/Agent-Ark/Toucan-1.5M"
        payload["resources"]["gpus_per_node"] = 4
        payload["resources"]["num_nodes"] = 2
    else:
        payload["resources"]["num_nodes"] = 1
    response = TestClient(
        create_app(
            FakeAuthenticator(),
            FakeAuthorizer(),
            submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    ).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=payload,
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 202
    assert response.json()["id"] == "pvc-training"


@pytest.mark.parametrize(
    "algorithm,parameters",
    [
        ("osft", {"unfreeze_rank_ratio": 0.1}),
        ("lora_grpo", {"tasks_per_iteration": 10}),
    ],
)
def test_http_post_rejects_invalid_algorithm_overrides(
    monkeypatch: pytest.MonkeyPatch, algorithm: str, parameters: dict[str, Any]
) -> None:
    _mock_sdk(monkeypatch)
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["algorithm"] = {"name": algorithm, "parameters": parameters}
    response = TestClient(
        create_app(
            FakeAuthenticator(),
            FakeAuthorizer(),
            submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    ).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=payload,
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert FakeRayJob.instances == []


@pytest.mark.parametrize(
    "change",
    [
        {"resources": {"num_nodes": 1}},
        {"resources": {"gpus_per_node": 0}},
        {"resources": {"gpu_type": "amd.com/gpu"}},
        {"algorithm": {"parameters": {"prompt_batch_size": 9, "group_size": 1}}},
        {"resources": {"num_nodes": 4, "gpus_per_node": 1}},
        {
            "resources": {"num_nodes": 2, "gpus_per_node": 8},
            "algorithm": {"parameters": {"prompt_batch_size": 1, "group_size": 32}},
        },
        {"algorithm": {"parameters": {"group_size": 4, "tensor_parallel_size": 3}}},
        {"algorithm": {"parameters": {"tensor_parallel_size": 0}}},
        {"algorithm": {"parameters": {"tensor_parallel_size": -1}}},
        {"output": {"destination_uri": "s3://checkpoint-storage/bucket/checkpoints/"}},
        {"model": {"uri": "s3://model-storage/bucket/model/"}},
        {"dataset": {"uri": "hf://other-secret/Agent-Ark/Toucan-1.5M"}},
        {"dataset": {"uri": "pvc://training-data/dataset/"}},
        {"algorithm": {"parameters": {"group_size": 4, "reward_type": "math"}}},
        {
            "algorithm": {"parameters": {"group_size": 4, "reward_type": "math"}},
            "dataset": {"uri": "pvc://training-data/dataset/train.json"},
        },
    ],
)
def test_invalid_grpo_layout_or_inputs_fail_before_sdk(change: dict[str, Any]) -> None:
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["algorithm"] = {"name": "lora_grpo", "parameters": {"group_size": 4}}
    payload["resources"]["num_nodes"] = 2
    payload["model"]["uri"] = "hf://hf-secret-token/Qwen/Qwen3-4B"
    payload["dataset"]["uri"] = "hf://hf-secret-token/Agent-Ark/Toucan-1.5M"
    for key, fields in change.items():
        payload[key].update(fields)
    body = CreateJobRequest.model_validate(payload)
    with pytest.raises(ApiError) as raised:
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
            "team-alpha", body, CallerIdentity("alice", None, ()), "caller-token"
        )
    assert raised.value.status_code == 422


def test_valid_grpo_batch_uses_total_gpu_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_sdk(monkeypatch)
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["algorithm"] = {
        "name": "lora_grpo",
        "parameters": {"group_size": 4, "prompt_batch_size": 12},
    }
    payload["resources"].update({"num_nodes": 4, "gpus_per_node": 1})
    payload["model"]["uri"] = "hf://hf-secret-token/Qwen/Qwen3-4B"
    payload["dataset"]["uri"] = "hf://hf-secret-token/Agent-Ark/Toucan-1.5M"
    job_id = submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        "team-alpha",
        CreateJobRequest.model_validate(payload),
        CallerIdentity("alice", None, ()),
        "caller-token",
    )
    assert job_id == payload["name"]
    assert FakeRayJob.instances[0].kwargs["cluster_config"].num_workers == 3


@pytest.mark.parametrize("algorithm", ["sft", "osft", "lora_sft", "lora_grpo"])
def test_entrypoint_selects_training_hub_algorithm_class_and_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, algorithm: str
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def record(name: str) -> Any:
        return lambda **kwargs: calls.append((name, kwargs))

    def factory(name: str, backend: str) -> Any:
        assert backend == _EXPECTED_TH_BACKENDS[name]
        return SimpleNamespace(train=record(name))

    monkeypatch.setitem(sys.modules, "training_hub", SimpleNamespace(create_algorithm=factory))
    monkeypatch.setattr(ray_entrypoint, "_dataset_format", lambda _path: "alpaca")
    parameters = {"lora_r": 16} if algorithm in {"lora_sft", "lora_grpo"} else {"num_epochs": 1}

    ray_entrypoint._run_training(algorithm, "/model", "/data.jsonl", tmp_path, parameters, 4, 1)

    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == algorithm
    assert kwargs["model_path"] == "/model"
    assert kwargs["data_path"] == "/data.jsonl"
    assert kwargs["ckpt_output_dir"] == str(tmp_path)
    assert kwargs["lora_r" if algorithm in {"lora_sft", "lora_grpo"} else "num_epochs"]
    if algorithm in {"sft", "osft"}:
        assert kwargs["data_output_dir"] == str(tmp_path / "data")
        assert kwargs["nproc_per_node"] == 4
        assert kwargs["nnodes"] == 1
    elif algorithm == "lora_sft":
        assert kwargs["dataset_type"] == "alpaca"
    else:
        assert "backend" not in kwargs
        assert kwargs["n_gpus"] == 4
        assert kwargs["nnodes"] == 1


def test_grpo_hf_inputs_remain_remote_for_worker_access(tmp_path: Path) -> None:
    location = {"scheme": "hf", "repository": "Agent-Ark/Toucan-1.5M"}
    assert ray_entrypoint._input(location, "dataset", tmp_path, remote_hf=True) == (
        "Agent-Ark/Toucan-1.5M"
    )


def test_grpo_trailing_slash_pvc_model_is_normalized_before_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: dict[str, Any] = {}
    monkeypatch.setitem(
        sys.modules,
        "training_hub",
        SimpleNamespace(
            create_algorithm=lambda _name, _backend: SimpleNamespace(
                train=lambda **kwargs: recorded.update(kwargs)
            )
        ),
    )
    model = tmp_path / "model"
    model.mkdir()
    ray_entrypoint.main(
        {
            "algorithm": "lora_grpo",
            "model": {"scheme": "pvc", "mount_path": str(model) + "/"},
            "dataset": {"scheme": "hf", "repository": "Agent-Ark/Toucan-1.5M"},
            "output": {"scheme": "pvc", "mount_path": str(tmp_path / "output")},
            "parameters": {},
            "gpus_per_node": 4,
            "num_workers": 1,
        }
    )
    assert recorded["model_path"] == str(model)


def test_hf_dataset_config_and_split_reach_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: dict[str, Any] = {}
    fake_dataset = SimpleNamespace(to_json=lambda path: Path(path).write_text("{}\n"))
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=lambda *args, **kwargs: recorded.update(kwargs) or fake_dataset
        ),
    )
    credentials = tmp_path / "credentials"
    credentials.mkdir()
    (credentials / "HF_TOKEN").write_text("test-token")
    location = {
        "scheme": "hf",
        "repository": "LipengCS/Table-GPT",
        "credential_dir": str(credentials),
        "token_key": "HF_TOKEN",
        "config": "All",
        "split": "train[:10]",
    }
    result = ray_entrypoint._input(location, "dataset", tmp_path)
    assert recorded == {"name": "All", "split": "train[:10]", "token": "test-token"}
    assert result == str(tmp_path / "dataset.jsonl")


@pytest.mark.parametrize("config", [None, "All"])
def test_hf_config_preflight_passes_secret_and_optional_config_to_datasets(
    monkeypatch: pytest.MonkeyPatch, config: str | None
) -> None:
    calls: list[tuple[str, str | None, str]] = []
    builder = SimpleNamespace(
        info=SimpleNamespace(splits={"train": SimpleNamespace(num_examples=100)})
    )
    monkeypatch.setattr(
        hf_dataset,
        "load_dataset_builder",
        lambda repository, *, name, token: calls.append((repository, name, token)) or builder,
    )

    hf_dataset.validate_hf_dataset(
        "LipengCS/Table-GPT", config, "train", "secret-sentinel", remote_loader=False
    )

    assert calls == [("LipengCS/Table-GPT", config, "secret-sentinel")]


def test_hf_preflight_returns_known_columns_without_loading_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = SimpleNamespace(
        info=SimpleNamespace(
            splits={"train": SimpleNamespace(num_examples=5)},
            features={"instruction": object(), "output": object()},
        )
    )
    monkeypatch.setattr(hf_dataset, "load_dataset_builder", lambda *_args, **_kwargs: builder)
    assert hf_dataset.validate_hf_dataset(
        "tatsu-lab/alpaca", None, "train", "secret", remote_loader=False
    ) == {"instruction", "output"}


@pytest.mark.parametrize(
    "algorithm,parameters,columns,expected",
    [
        ("sft", {}, {"text"}, "messages"),
        ("osft", {}, {"text"}, "messages"),
        ("sft", {"is_pretraining": True}, {"text"}, "document"),
        ("osft", {"is_pretraining": True}, {"text"}, "document"),
        ("sft", {"is_pretraining": True, "document_column_name": None}, {"text"}, "document"),
        ("osft", {"is_pretraining": True, "document_column_name": None}, {"text"}, "document"),
        ("sft", {"is_pretraining": True, "document_column_name": "document"}, {"text"}, "document"),
        ("lora_sft", {"dataset_type": "alpaca"}, {"question", "answer"}, None),
        ("lora_sft", {}, {"instruction", "output"}, None),
        ("lora_sft", {}, {"image", "caption"}, None),
        ("lora_sft", {"dataset_type": "passthrough"}, {"text"}, None),
        ("osft", {"use_processed_dataset": True}, {"input_ids"}, None),
        ("lora_grpo", {}, {"messages"}, None),
    ],
)
def test_dataset_columns_follow_training_hub_backend(
    algorithm: str, parameters: dict[str, Any], columns: set[str], expected: str | None
) -> None:
    if expected is None:
        validate_dataset_columns(algorithm, parameters, columns)
    else:
        with pytest.raises(ApiError) as caught:
            validate_dataset_columns(algorithm, parameters, columns)
        assert caught.value.status_code == 422
        assert expected in caught.value.message


def test_unknown_hf_columns_do_not_reject_submission() -> None:
    validate_dataset_columns("sft", {}, None)
    validate_dataset_columns("sft", {}, set())


@pytest.mark.parametrize(
    "path", ["train.jsonl", "train.json", "train.csv", "train.parquet", "data/"]
)
def test_s3_dataset_supported_file_shapes(path: str) -> None:
    validate_dataset_location({"scheme": "s3", "path": path})


@pytest.mark.parametrize("suffix", [".jsonl", ".json", ".csv", ".parquet"])
def test_s3_prefix_selects_any_training_hub_file_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str
) -> None:
    data_file = tmp_path / f"train{suffix}"
    data_file.write_bytes(b"sample")
    monkeypatch.setattr(ray_entrypoint, "_download_s3", lambda *_args: str(tmp_path))
    assert ray_entrypoint._input(
        {"scheme": "s3", "bucket": "bucket", "path": "datasets/"}, "dataset", tmp_path
    ) == str(data_file)


def test_s3_prefix_rejects_ambiguous_data_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "train.csv").write_bytes(b"sample")
    (tmp_path / "train.parquet").write_bytes(b"sample")
    monkeypatch.setattr(ray_entrypoint, "_download_s3", lambda *_args: str(tmp_path))
    with pytest.raises(ValueError, match="exactly one supported data file"):
        ray_entrypoint._input(
            {"scheme": "s3", "bucket": "bucket", "path": "datasets/"}, "dataset", tmp_path
        )


def test_s3_dataset_rejects_unreadable_object_type_before_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["dataset"]["uri"] = "s3://training-data/bucket/train.txt"
    request = CreateJobRequest.model_validate(payload)
    with pytest.raises(ApiError) as caught:
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
            "team-alpha", request, CallerIdentity("alice", None, ()), "caller-token"
        )
    assert caught.value.status_code == 422
    assert "JSONL, JSON, CSV, or Parquet" in caught.value.message
    assert not FakeRayJob.instances


def test_hf_known_columns_reject_before_rayjob_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    monkeypatch.setattr(submission, "validate_hf_dataset", lambda *_args, **_kwargs: {"text"})
    payload = _sample("huggingface_inputs_s3_output").model_dump(mode="json", exclude_unset=True)
    payload["algorithm"] = {"name": "sft", "parameters": {"num_epochs": 1}}
    request = CreateJobRequest.model_validate(payload)
    with pytest.raises(ApiError) as caught:
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
            "team-alpha",
            request,
            CallerIdentity("alice", None, ()),
            "caller-token",
        )
    assert caught.value.status_code == 422
    assert "messages" in caught.value.message
    assert not FakeRayJob.instances


@pytest.mark.parametrize("config", [None, "not-a-config"])
def test_hf_config_preflight_rejects_missing_or_invalid_config_without_leaking_token(
    monkeypatch: pytest.MonkeyPatch, config: str | None
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        reason = "Config name is missing" if config is None else "BuilderConfig not found"
        raise ValueError(f"{reason}: secret-sentinel must not be exposed")

    monkeypatch.setattr(hf_dataset, "load_dataset_builder", fail)

    with pytest.raises(ApiError) as caught:
        hf_dataset.validate_hf_dataset(
            "LipengCS/Table-GPT", config, "train", "secret-sentinel", remote_loader=False
        )

    assert caught.value.status_code == 422
    assert caught.value.code == "validation_error"
    assert "dataset.config" in caught.value.message
    assert "secret-sentinel" not in caught.value.message


def test_hf_config_preflight_maps_remote_failure_without_leaking_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ConnectionError("secret-sentinel must not be exposed")

    monkeypatch.setattr(hf_dataset, "load_dataset_builder", fail)

    with pytest.raises(ApiError) as caught:
        hf_dataset.validate_hf_dataset(
            "tatsu-lab/alpaca", None, "train", "secret-sentinel", remote_loader=False
        )

    assert caught.value.status_code == 503
    assert caught.value.code == "submission_unavailable"
    assert "secret-sentinel" not in caught.value.message


def test_hf_dataset_value_error_does_not_mislabel_unrelated_failure_as_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("invalid data files: secret-sentinel")

    monkeypatch.setattr(hf_dataset, "load_dataset_builder", fail)

    with pytest.raises(ApiError) as caught:
        hf_dataset.validate_hf_dataset(
            "my-org/data", None, "train", "secret-sentinel", remote_loader=False
        )

    assert caught.value.status_code == 422
    assert "repository files" in caught.value.message
    assert "dataset.config" not in caught.value.message
    assert "secret-sentinel" not in caught.value.message


def test_hf_dataset_preflight_rejects_inaccessible_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise DatasetNotFoundError("secret-sentinel must not be exposed")

    monkeypatch.setattr(hf_dataset, "load_dataset_builder", fail)

    with pytest.raises(ApiError) as caught:
        hf_dataset.validate_hf_dataset(
            "missing/data", None, "train", "secret-sentinel", remote_loader=False
        )

    assert caught.value.status_code == 422
    assert "dataset.uri" in caught.value.message
    assert "Secret token" in caught.value.message
    assert "secret-sentinel" not in caught.value.message


@pytest.mark.parametrize("split", ["unknown", "train[bad]"])
def test_hf_dataset_preflight_rejects_invalid_split(
    monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    builder = SimpleNamespace(
        info=SimpleNamespace(splits={"train": SimpleNamespace(num_examples=100)})
    )
    monkeypatch.setattr(hf_dataset, "load_dataset_builder", lambda *_args, **_kwargs: builder)

    with pytest.raises(ApiError) as caught:
        hf_dataset.validate_hf_dataset(
            "tatsu-lab/alpaca", None, split, "secret", remote_loader=False
        )

    assert caught.value.status_code == 422
    assert "dataset.split" in caught.value.message


def test_hf_model_preflight_rejects_inaccessible_repository_without_leaking_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "https://huggingface.co/api/models/missing/model")
    response = httpx.Response(404, request=request)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RepositoryNotFoundError("secret-sentinel must not be exposed", response=response)

    monkeypatch.setattr(hf_dataset, "HfApi", lambda: SimpleNamespace(model_info=fail))

    with pytest.raises(ApiError) as caught:
        hf_dataset.validate_hf_model("missing/model", "secret-sentinel")

    assert caught.value.status_code == 422
    assert "model.uri" in caught.value.message
    assert "token permissions" in caught.value.message
    assert "secret-sentinel" not in caught.value.message


def test_remote_grpo_dataset_checks_repository_without_dataset_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def dataset_info(repository: str, *, token: str, timeout: int) -> None:
        assert timeout == 10
        calls.append((repository, token))

    monkeypatch.setattr(hf_dataset, "HfApi", lambda: SimpleNamespace(dataset_info=dataset_info))
    monkeypatch.setattr(
        hf_dataset,
        "load_dataset_builder",
        lambda *_args, **_kwargs: pytest.fail("GRPO does not use datasets.load_dataset"),
    )

    hf_dataset.validate_hf_dataset(
        "Agent-Ark/Toucan-1.5M", None, "train", "secret-sentinel", remote_loader=True
    )

    assert calls == [("Agent-Ark/Toucan-1.5M", "secret-sentinel")]


def test_hf_config_preflight_rejects_before_rayjob_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    observed: list[tuple[str, str | None, str, str]] = []

    def reject(
        repository: str, config: str | None, split: str, token: str, *, remote_loader: bool
    ) -> None:
        assert not remote_loader
        observed.append((repository, config, split, token))
        raise ApiError(422, "validation_error", "dataset.config is required.")

    monkeypatch.setattr(submission, "validate_hf_dataset", reject)
    request = _sample("huggingface_inputs_s3_output")

    with pytest.raises(ApiError, match="dataset.config is required"):
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
            "team-alpha", request, CallerIdentity("alice", None, ()), "caller-token"
        )

    assert observed == [("tatsu-lab/alpaca", None, "train", "secret-sentinel")]
    assert not FakeRayJob.instances


def test_hf_model_preflight_rejects_before_dataset_check_and_rayjob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    observed: list[tuple[str, str]] = []

    def reject(repository: str, token: str) -> None:
        observed.append((repository, token))
        raise ApiError(422, "validation_error", "model.uri is inaccessible.")

    monkeypatch.setattr(submission, "validate_hf_model", reject)
    monkeypatch.setattr(
        submission,
        "validate_hf_dataset",
        lambda *_args, **_kwargs: pytest.fail("dataset check must not run after model failure"),
    )

    with pytest.raises(ApiError, match="model.uri is inaccessible"):
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
            "team-alpha",
            _sample("huggingface_inputs_s3_output"),
            CallerIdentity("alice", None, ()),
            "caller-token",
        )

    assert observed == [("meta-llama/Llama-3.2-3B", "secret-sentinel")]
    assert not FakeRayJob.instances


def test_hf_dataset_options_are_preserved_in_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    configuration: dict[str, Any] = {}
    monkeypatch.setattr(
        submission, "_entrypoint", lambda value: configuration.update(value) or "python -c 'pass'"
    )
    payload = _sample("huggingface_inputs_s3_output").model_dump(mode="json", exclude_unset=True)
    payload["dataset"] = {
        "uri": "hf://hf-secret-token/LipengCS/Table-GPT",
        "config": "All",
        "split": "train[:10]",
    }
    request = CreateJobRequest.model_validate(payload)
    submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        "team-alpha", request, CallerIdentity("alice", None, ()), "caller-token"
    )
    assert configuration["dataset"]["repository"] == "LipengCS/Table-GPT"
    assert configuration["dataset"]["config"] == "All"
    assert configuration["dataset"]["split"] == "train[:10]"


@pytest.mark.parametrize("dataset", [{"config": "All"}, {"split": "train"}])
def test_hf_dataset_options_reject_non_hf_and_remote_grpo(
    dataset: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_sdk(monkeypatch)
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["dataset"].update(dataset)
    with pytest.raises(ApiError, match="Hugging Face dataset URI"):
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
            "team-alpha",
            CreateJobRequest.model_validate(payload),
            CallerIdentity("alice", None, ()),
            "caller-token",
        )
    payload["algorithm"] = {"name": "lora_grpo"}
    payload["resources"]["num_nodes"] = 2
    payload["dataset"]["uri"] = "hf://hf-secret-token/Agent-Ark/Toucan-1.5M"
    with pytest.raises(ApiError, match="remote GRPO loader"):
        submission.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
            "team-alpha",
            CreateJobRequest.model_validate(payload),
            CallerIdentity("alice", None, ()),
            "caller-token",
        )
    assert FakeRayJob.instances == []


def test_s3_dependency_preflight_skips_install_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ray_entrypoint, "_s3_modules_available", lambda: True)
    monkeypatch.setattr(
        ray_entrypoint.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("S3 packages must not be reinstalled"),
    )
    ray_entrypoint._ensure_s3_runtime(tmp_path)


def test_s3_dependency_preflight_installs_pinned_pair_and_waits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checks = iter((False, True))
    monkeypatch.setattr(ray_entrypoint, "_s3_modules_available", lambda: next(checks))
    commands: list[list[str]] = []

    def install(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        assert kwargs["timeout"] == 180
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ray_entrypoint.subprocess, "run", install)
    previous_path = list(sys.path)
    try:
        ray_entrypoint._ensure_s3_runtime(tmp_path)
        assert sys.path[0] == str(tmp_path / "site-packages")
    finally:
        sys.path[:] = previous_path
    assert len(commands) == 1
    assert "boto3==1.40.61" in commands[0]
    assert "botocore==1.40.61" in commands[0]


def test_s3_dependency_preflight_fails_before_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ray_entrypoint, "_s3_modules_available", lambda: False)
    monkeypatch.setattr(
        ray_entrypoint.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(RuntimeError, match="Check pip and package index or mirror access"):
        ray_entrypoint._ensure_s3_runtime(tmp_path)


def test_s3_dependency_failure_stops_before_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_preflight(_workdir: Path) -> None:
        raise RuntimeError("S3 support is unavailable")

    monkeypatch.setattr(ray_entrypoint, "_ensure_s3_runtime", fail_preflight)
    monkeypatch.setattr(
        ray_entrypoint,
        "_run_training",
        lambda *_args: pytest.fail("Training must not start without S3 support"),
    )
    with pytest.raises(RuntimeError, match="S3 support is unavailable"):
        ray_entrypoint.main(
            {
                "algorithm": "lora_sft",
                "model": {"scheme": "pvc", "mount_path": "/mnt/model"},
                "dataset": {"scheme": "pvc", "mount_path": "/mnt/train.jsonl"},
                "output": {"scheme": "s3", "path": "checkpoints", "bucket": "bucket"},
                "parameters": {},
                "gpus_per_node": 1,
                "num_workers": 0,
            }
        )


def test_s3_output_preflight_runs_before_training(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        ray_entrypoint, "_ensure_s3_runtime", lambda _workdir: events.append("preflight")
    )
    monkeypatch.setattr(ray_entrypoint, "_run_training", lambda *_args: events.append("training"))
    monkeypatch.setattr(ray_entrypoint, "_upload_s3", lambda *_args: events.append("upload"))
    ray_entrypoint.main(
        {
            "algorithm": "lora_sft",
            "model": {"scheme": "pvc", "mount_path": "/mnt/model"},
            "dataset": {"scheme": "pvc", "mount_path": "/mnt/train.jsonl"},
            "output": {"scheme": "s3", "path": "checkpoints", "bucket": "bucket"},
            "parameters": {},
            "gpus_per_node": 1,
            "num_workers": 0,
        }
    )
    assert events == ["preflight", "training", "upload"]


def test_s3_checkpoint_upload_uses_expected_object_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "weights.safetensors").write_bytes(b"weights")
    uploads: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        ray_entrypoint,
        "_s3_client",
        lambda _location: SimpleNamespace(
            upload_file=lambda path, bucket, key: uploads.append((path, bucket, key))
        ),
    )
    ray_entrypoint._upload_s3(
        {"scheme": "s3", "bucket": "model-bucket", "path": "checkpoints/run-1/"}, output
    )
    assert uploads == [
        (
            str(adapter / "weights.safetensors"),
            "model-bucket",
            "checkpoints/run-1/adapter/weights.safetensors",
        )
    ]


@pytest.mark.parametrize("algorithm", ["sft", "lora_grpo"])
def test_embedded_ray_entrypoint_executes_the_selected_algorithm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, algorithm: str
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def record(name: str) -> Any:
        return lambda **kwargs: calls.append((name, kwargs))

    def factory(name: str, backend: str) -> Any:
        assert backend == _EXPECTED_TH_BACKENDS[name]
        return SimpleNamespace(train=record(name))

    monkeypatch.setitem(sys.modules, "training_hub", SimpleNamespace(create_algorithm=factory))
    model = tmp_path / "model"
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")
    if algorithm == "lora_grpo":
        model_location = {"scheme": "hf", "repository": "Qwen/Qwen3-4B"}
        dataset_location = {"scheme": "hf", "repository": "Agent-Ark/Toucan-1.5M"}
    else:
        model_location = {"scheme": "pvc", "mount_path": str(model)}
        dataset_location = {"scheme": "pvc", "mount_path": str(dataset)}
    configuration = {
        "algorithm": algorithm,
        "model": model_location,
        "dataset": dataset_location,
        "output": {"scheme": "pvc", "mount_path": str(tmp_path / "checkpoints")},
        "parameters": {"num_iterations": 2} if algorithm == "lora_grpo" else {"num_epochs": 1},
        "gpus_per_node": 4 if algorithm == "lora_grpo" else 1,
        "num_workers": 1 if algorithm == "lora_grpo" else 0,
    }

    command = shlex.split(submission._entrypoint(configuration))
    assert command[:2] == ["python", "-c"]
    exec(command[2], {})

    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == algorithm
    assert kwargs["ckpt_output_dir"] == str(tmp_path / "checkpoints")
    if algorithm == "lora_grpo":
        assert kwargs["model_path"] == "Qwen/Qwen3-4B"
        assert kwargs["data_path"] == "Agent-Ark/Toucan-1.5M"
        assert kwargs["num_iterations"] == 2
        assert kwargs["n_gpus"] == 4
    else:
        assert kwargs["model_path"] == str(model)
        assert kwargs["data_path"] == str(dataset)
        assert kwargs["num_epochs"] == 1
