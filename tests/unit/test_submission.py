"""SDK-mocked submission tests; no Ray or Kubernetes cluster is contacted."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import pytest
from codeflare_sdk import RayJob
from codeflare_sdk.vendored.python_client.kuberay_job_api import RayjobApi
from fastapi.testclient import TestClient
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from training_service import job_submission as submission
from training_service import jobs
from training_service.auth import CallerIdentity
from training_service.authorization import ProjectPermission
from training_service.contract import load_contract
from training_service.errors import ApiError
from training_service.main import create_app
from training_service.ray_entrypoint import _dataset_format
from training_service_api.models.create_job_request import CreateJobRequest
from training_service_api.models.error_response import ErrorResponse


def _sample(name: str = "pvc_inputs_output") -> CreateJobRequest:
    examples = load_contract()["paths"]["/projects/{project}/training-jobs"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]
    return CreateJobRequest.model_validate(examples[name]["value"])


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
def test_advanced_credential_and_algorithm_fields_are_out_of_basic_scope(
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
            submission.CodeFlareJobSubmitter("quay.io/example/ray:2.55.1"),
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
            submission.CodeFlareJobSubmitter(""),
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

    def list_jobs(request: Any, project: str, *args: Any) -> dict[str, Any]:
        calls.append((request.state.caller_identity.username, project))
        return {"items": [], "total_count": 0}

    monkeypatch.setattr(jobs, "list_jobs", list_jobs)
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
        return SimpleNamespace(data=dict.fromkeys(keys, "opaque-value"), string_data=None)


class FakeRayJob:
    instances: list[FakeRayJob] = []
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.instances.append(self)

    def submit(self) -> str:
        self.client_at_submit = submission.codeflare_auth.api_client
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
    monkeypatch.setattr(submission.codeflare_auth, "api_client", None)
    monkeypatch.setattr(submission.codeflare_auth, "config_path", None)

    def set_api_client(api_client: FakeApiClient) -> None:
        submission.codeflare_auth.api_client = api_client
        submission.codeflare_auth.config_path = "custom"

    monkeypatch.setattr(submission.codeflare_auth, "set_api_client", set_api_client)
    return core


def test_pvc_submission_maps_resources_and_retains_only_safe_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _mock_sdk(monkeypatch)
    body = _sample()
    identifier = submission.CodeFlareJobSubmitter("quay.io/example/ray:2.55.1")._submit_sync(
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
    assert "api_client" not in kwargs
    assert FakeRayJob.instances[0].client_at_submit.token == "caller-token"
    assert submission.codeflare_auth.api_client is None
    assert submission.codeflare_auth.config_path is None
    cluster = kwargs["cluster_config"]
    assert cluster.image == "quay.io/example/ray:2.55.1"
    assert cluster.num_workers == 1
    assert cluster.head_accelerators == {"nvidia.com/gpu": 4}
    assert cluster.worker_accelerators == {"nvidia.com/gpu": 4}
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
    submitter = submission.CodeFlareJobSubmitter("quay.io/example/ray:2.55.1")

    with pytest.raises(ApiError) as raised:
        submitter._submit_sync("team-alpha", body, CallerIdentity("alice", None, ()), "token")
    assert raised.value.status_code == 422
    assert raised.value.code == "validation_error"
    assert FakeRayJob.instances == []

    body.resources.num_nodes = 1
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
    image = "quay.io/example/ray:2.55.1"
    _mock_sdk(monkeypatch)
    submission.CodeFlareJobSubmitter(image)._submit_sync(
        "team-alpha", _sample(), CallerIdentity("alice", None, ()), "caller-token"
    )
    cluster = FakeRayJob.instances[0].kwargs["cluster_config"]
    assert cluster.image == image
    spec = cluster.build_ray_cluster_spec("pvc-training-cluster")
    assert spec["rayVersion"] == "2.55.1"
    assert spec["headGroupSpec"]["template"].spec.containers[0].image == image
    assert spec["workerGroupSpecs"][0]["template"].spec.containers[0].image == image


@pytest.mark.parametrize("num_nodes", [1, 2])
def test_sdk_submission_metadata_and_topology_are_readable(
    monkeypatch: pytest.MonkeyPatch, num_nodes: int
) -> None:
    core = _mock_sdk(monkeypatch)
    payload = _sample().model_dump(mode="json", exclude_unset=True)
    payload["resources"]["num_nodes"] = num_nodes
    authorizer = FakeAuthorizer()
    http = TestClient(
        create_app(
            FakeAuthenticator(),
            authorizer,
            submission.CodeFlareJobSubmitter("quay.io/example/ray:2.55.1"),
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
            kwargs["cluster_config"].build_ray_cluster_spec("pvc-training-cluster")
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
    assert response.json()["resources"] == payload["resources"]
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

    submission.CodeFlareJobSubmitter("quay.io/example/ray:2.55.1")._submit_sync(
        "team-alpha", body, CallerIdentity("alice", None, ()), "caller-token"
    )

    kwargs = FakeRayJob.instances[0].kwargs
    cluster = kwargs["cluster_config"]
    assert kwargs["local_queue"] is None
    assert cluster.num_workers == 0
    assert cluster.head_cpu_requests == "8"
    assert cluster.head_memory_requests == "32Gi"
    assert cluster.head_accelerators == {"nvidia.com/gpu": 1}
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
    submission.CodeFlareJobSubmitter("quay.io/example/ray:2.55.1")._submit_sync(
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
    submitter = submission.CodeFlareJobSubmitter("quay.io/example/ray:2.55.1")
    with pytest.raises(ApiError) as raised:
        await submitter.submit(
            "team-alpha", _sample(), CallerIdentity("alice", None, ()), "caller-token"
        )
    assert raised.value.status_code == 409
    assert submission.codeflare_auth.api_client is None
    assert submission.codeflare_auth.config_path is None


def test_sdk_auth_is_restored_when_client_verification_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    previous_client = FakeApiClient("previous-caller")
    submission.codeflare_auth.api_client = previous_client
    submission.codeflare_auth.config_path = "previous-config"

    def reject_client(api_client: FakeApiClient) -> None:
        submission.codeflare_auth.api_client = api_client
        submission.codeflare_auth.config_path = "custom"
        raise ApiException(status=401, reason="Invalid caller token")

    monkeypatch.setattr(submission.codeflare_auth, "set_api_client", reject_client)
    with pytest.raises(ApiException) as raised:
        with submission._sdk_client_scope(FakeApiClient("invalid-caller")):
            pytest.fail("SDK scope must not be entered after verification fails")

    assert raised.value.status == 401
    assert submission.codeflare_auth.api_client is previous_client
    assert submission.codeflare_auth.config_path == "previous-config"


def test_parallel_submissions_keep_each_callers_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sdk(monkeypatch)
    first_submitting = Event()
    second_prepared = Event()
    release_first = Event()
    preparation_lock = Lock()
    prepared = 0
    original_storage_plan = submission._storage_plan

    def storage_plan(*args: Any, **kwargs: Any) -> Any:
        nonlocal prepared
        result = original_storage_plan(*args, **kwargs)
        with preparation_lock:
            prepared += 1
            if prepared == 2:
                second_prepared.set()
        return result

    def controlled_submit(job: FakeRayJob) -> str:
        job.client_at_submit = submission.codeflare_auth.api_client
        if job.client_at_submit.token == "caller-a":
            first_submitting.set()
            assert release_first.wait(5)
            assert submission.codeflare_auth.api_client is job.client_at_submit
        return str(job.kwargs["job_name"])

    monkeypatch.setattr(submission, "_storage_plan", storage_plan)
    monkeypatch.setattr(FakeRayJob, "submit", controlled_submit)
    submitters = [submission.CodeFlareJobSubmitter("quay.io/example/ray:2.55.1") for _ in range(2)]

    def submit(index: int, name: str, token: str) -> str:
        body = _sample()
        body.name = name
        return submitters[index]._submit_sync(
            "team-alpha", body, CallerIdentity(name, None, ()), token
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(submit, 0, "job-a", "caller-a")
        try:
            assert first_submitting.wait(5)
            second = pool.submit(submit, 1, "job-b", "caller-b")
            assert second_prepared.wait(5)
            assert submission.codeflare_auth.api_client.token == "caller-a"
            assert not second.done()
        finally:
            release_first.set()
        assert first.result(timeout=5) == "job-a"
        assert second.result(timeout=5) == "job-b"

    assert [job.client_at_submit.token for job in FakeRayJob.instances] == [
        "caller-a",
        "caller-b",
    ]
    assert submission.codeflare_auth.api_client is None
    assert submission.codeflare_auth.config_path is None


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
    assert submission._map_backend_error(raised.value).status_code == api_status

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
    observed: dict[str, Any] = {}

    def fake_incluster_config(*, client_configuration: Any, try_refresh_token: bool) -> None:
        observed["refresh"] = try_refresh_token
        client_configuration.api_key["authorization"] = "bearer service-account-token"

    def fake_api_client(configuration: Any) -> Any:
        observed["configuration"] = configuration
        return configuration

    monkeypatch.setattr(submission.config, "load_incluster_config", fake_incluster_config)
    monkeypatch.setattr(submission, "_CallerApiClient", fake_api_client)
    result = submission._caller_api_client("caller-token")
    assert observed["refresh"] is False
    assert result.api_key["authorization"] == "Bearer caller-token"
    assert "service-account-token" not in result.api_key["authorization"]


@pytest.mark.parametrize(
    "backend_status,api_status",
    [(401, 401), (403, 403), (404, 404), (409, 409), (422, 422), (500, 503)],
)
def test_kubernetes_errors_map_to_documented_statuses(backend_status: int, api_status: int) -> None:
    error = submission._map_backend_error(ApiException(status=backend_status))
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
