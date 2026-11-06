"""Project-scoped job reads/deletion with mocked Kubernetes and Ray SDKs."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from kubernetes.client.exceptions import ApiException

from training_service import jobs, kubernetes_client, logs
from training_service.auth import CallerIdentity
from training_service.errors import ApiError
from training_service.main import create_app

BASE = "/api/v1/projects/team-a/training-jobs"
HEADERS = {"Authorization": "Bearer unit-test-token"}
PAYLOAD = {
    "name": "example",
    "model": {"uri": "pvc://inputs/model"},
    "dataset": {"uri": "pvc://inputs/train.jsonl"},
    "algorithm": {"name": "lora_sft", "parameters": {"learning_rate": 0.0001}},
}


class Authenticator:
    async def authenticate(self, token):
        return CallerIdentity("alice", "uid-a", ("developers",))


class Authorizer:
    def __init__(self):
        self.allowed = True
        self.calls = []

    async def is_allowed(self, identity, project, permission):
        self.calls.append((project, permission))
        return self.allowed


def job(name="example", status="RUNNING"):
    return {
        "metadata": {
            "name": name,
            "uid": "job-uid",
            "resourceVersion": "42",
            "creationTimestamp": "2026-10-21T09:00:00Z",
            "annotations": {
                "training.rhoai.io/submitted-by": "original-submitter",
                "training.rhoai.io/submission": json.dumps(PAYLOAD),
            },
        },
        "spec": {},
        "status": {
            "jobStatus": status,
            "rayClusterName": "example-cluster",
            "jobId": "raysubmit_123",
            "startTime": 1792573200000,
            "endTime": 1792573260000,
        },
    }


@pytest.fixture
def backend(monkeypatch):
    custom, core, sdk = Mock(), Mock(), Mock()
    record = job()
    cluster = {
        "metadata": {
            "ownerReferences": [
                {
                    "apiVersion": "ray.io/v1",
                    "kind": "RayJob",
                    "name": "example",
                    "uid": "job-uid",
                    "controller": True,
                }
            ]
        }
    }

    def read(group, version, namespace, resource, name, **kwargs):
        assert namespace == "team-a"
        return record if resource == "rayjobs" else cluster

    custom.get_namespaced_custom_object.side_effect = read
    custom.list_namespaced_custom_object.return_value = {"items": [record]}
    monkeypatch.setattr(kubernetes_client.client, "CustomObjectsApi", Mock(return_value=custom))
    monkeypatch.setattr(kubernetes_client.client, "CoreV1Api", Mock(return_value=core))
    sdk.get_job_logs.return_value = "first\nsecond\nthird\n"
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(jobs, "CodeFlareLogClient", factory)
    monkeypatch.setenv("TRAINING_SERVICE_RAY_GATEWAY_URL", "https://ray.example.com")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_API_SERVER_ADDRESS", raising=False)
    authorizer = Authorizer()
    http = TestClient(create_app(Authenticator(), authorizer))
    return http, custom, core, sdk, record, cluster, authorizer, factory


def test_list_is_compact_and_counts_all_kubernetes_pages(backend):
    http, custom, _, _, record, *_ = backend
    queued = job("queued", "PENDING")
    queued["metadata"]["labels"] = {"kueue.x-k8s.io/queue-name": "team-queue"}
    custom.list_namespaced_custom_object.side_effect = [
        {"items": [record], "metadata": {"continue": "kubernetes-next"}},
        {"items": [queued, job("completed", "SUCCEEDED")]},
    ]
    response = http.get(BASE + "?limit=1", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 3
    assert len(data["items"]) == 1
    assert set(data["items"][0]) == {"id", "status", "created_at"}
    assert data["items"][0]["id"] == "completed"
    assert custom.list_namespaced_custom_object.call_args.kwargs["_continue"] == "kubernetes-next"


def test_pagination_rechecks_access_and_binds_filters(backend):
    http, custom, _, _, _, _, authorizer, _ = backend
    custom.list_namespaced_custom_object.return_value = {"items": [job("a"), job("b")]}
    first = http.get(BASE + "?limit=1", headers=HEADERS).json()
    token = first["next_page_token"]
    second = http.get(BASE, params={"limit": 1, "page_token": token}, headers=HEADERS)
    assert second.status_code == 200
    assert second.json()["items"][0]["id"] == "b"
    assert second.json()["total_count"] == 2
    assert "next_page_token" not in second.json()
    assert (
        http.get(
            BASE, params={"page_token": token, "status": "FAILED"}, headers=HEADERS
        ).status_code
        == 400
    )
    authorizer.allowed = False
    custom.reset_mock()
    assert http.get(BASE, params={"page_token": token}, headers=HEADERS).status_code == 403
    custom.list_namespaced_custom_object.assert_not_called()


def test_list_filters_and_empty_project(backend):
    http, custom, *_ = backend
    queued = job("queued", "PENDING")
    queued["metadata"]["labels"] = {"kueue.x-k8s.io/queue-name": "team-queue"}
    custom.list_namespaced_custom_object.return_value = {"items": [job(), queued]}
    response = http.get(BASE + "?status=PENDING&queue_name=team-queue", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["total_count"] == 1
    assert response.json()["items"][0]["queue_name"] == "team-queue"
    custom.list_namespaced_custom_object.return_value = {"items": []}
    assert http.get(BASE, headers=HEADERS).json() == {"items": [], "total_count": 0}


def test_detail_preserves_submission_and_submitter_not_reader(backend):
    http, _, _, _, record, *_ = backend
    record["status"]["jobStatus"] = "SUCCEEDED"
    record["spec"]["runtimeEnv"] = {"env_vars": {"TOKEN": "must-not-leak"}}
    response = http.get(BASE + "/example", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["submission"] == PAYLOAD
    assert data["submitted_by"] == "original-submitter"
    assert data["finished_at"]
    assert data["started_at"]
    assert "must-not-leak" not in response.text
    assert "resources" not in data["submission"]


def test_external_and_pending_job_omit_unavailable_details(backend):
    http, _, _, _, record, *_ = backend
    record["metadata"].pop("annotations")
    record["status"] = {"jobDeploymentStatus": "Waiting"}
    data = http.get(BASE + "/example", headers=HEADERS).json()
    assert data["status"] == "PENDING"
    assert data["created_at"]
    assert not {"submission", "submitted_by", "finished_at", "started_at", "metrics"} & data.keys()
    record["metadata"]["annotations"] = {"training.rhoai.io/submission": '{"token":"invalid"}'}
    assert "submission" not in http.get(BASE + "/example", headers=HEADERS).json()


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "", "UNKNOWN"])
def test_active_or_unknown_job_cannot_be_deleted(backend, state):
    http, custom, _, _, record, *_ = backend
    record["status"].update(jobStatus=state, jobDeploymentStatus="Failed")
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 409
    assert response.json()["code"] == "job_not_terminal"
    custom.delete_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("state", ["SUCCEEDED", "FAILED", "STOPPED"])
def test_terminal_delete_uses_foreground_and_race_preconditions(backend, state):
    http, custom, _, sdk, record, *_ = backend
    record["status"]["jobStatus"] = state
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 204
    assert response.content == b""
    call = custom.delete_namespaced_custom_object.call_args
    assert call.args == ("ray.io", "v1", "team-a", "rayjobs", "example")
    options = call.kwargs["body"]
    assert options.propagation_policy == "Foreground"
    assert options.preconditions.uid == "job-uid"
    assert options.preconditions.resource_version == "42"
    sdk.stop_job.assert_not_called()
    assert custom.delete_namespaced_custom_object.call_count == 1


@pytest.mark.parametrize("case", ["shared", "unowned", "retrying"])
def test_deletion_never_cleans_shared_unowned_or_retrying_clusters(backend, case):
    http, custom, _, _, record, cluster, *_ = backend
    record["status"]["jobStatus"] = "SUCCEEDED"
    if case == "shared":
        record["spec"]["clusterSelector"] = {"ray.io/cluster": "shared"}
    elif case == "retrying":
        record["status"]["jobDeploymentStatus"] = "Retrying"
    else:
        cluster["metadata"]["ownerReferences"][0]["uid"] = "another-job"
    assert http.delete(BASE + "/example", headers=HEADERS).status_code == 409
    custom.delete_namespaced_custom_object.assert_not_called()


def test_already_removed_cluster_does_not_block_terminal_record_cleanup(backend):
    http, custom, _, _, record, *_ = backend
    record["status"]["jobStatus"] = "SUCCEEDED"
    custom.get_namespaced_custom_object.side_effect = [record, ApiException(status=404)]
    assert http.delete(BASE + "/example", headers=HEADERS).status_code == 204


@pytest.mark.parametrize("status", [401, 403, 404, 409, 500])
def test_backend_errors_are_sanitized(backend, status):
    http, custom, *_ = backend
    custom.get_namespaced_custom_object.side_effect = ApiException(
        status=status, reason="sensitive-backend-body"
    )
    response = http.get(BASE + "/example", headers=HEADERS)
    assert response.status_code == (status if status < 500 else 503)
    assert set(response.json()) == {"code", "message"}
    assert "sensitive-backend-body" not in response.text


@pytest.mark.parametrize(
    "method,suffix",
    [
        ("GET", "/example"),
        ("DELETE", "/example"),
        ("GET", "/example/logs?follow=false"),
        ("GET", "/example/logs?follow=true"),
    ],
)
def test_missing_job_errors_identify_job_and_project(backend, method, suffix):
    http, custom, _, _, _, _, _, factory = backend
    error = ApiException(status=404, reason="sensitive-backend-body")
    error.body = "unit-test-token"
    custom.get_namespaced_custom_object.side_effect = error
    response = http.request(method, BASE + suffix, headers=HEADERS)
    assert response.status_code == 404
    assert response.json() == {
        "code": "not_found",
        "message": "Training job 'example' was not found in project 'team-a'.",
    }
    assert response.headers["content-type"].startswith("application/json")
    assert "sensitive-backend-body" not in response.text
    assert "unit-test-token" not in response.text
    custom.delete_namespaced_custom_object.assert_not_called()
    factory.assert_not_called()


def test_job_disappearing_during_delete_has_context(backend):
    http, custom, _, _, record, *_ = backend
    record["status"]["jobStatus"] = "SUCCEEDED"
    custom.delete_namespaced_custom_object.side_effect = ApiException(status=404)
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 404
    assert response.json() == {
        "code": "not_found",
        "message": "Training job 'example' was not found in project 'team-a'.",
    }
    assert custom.delete_namespaced_custom_object.call_count == 1


@pytest.mark.parametrize(
    "method, suffix",
    [("GET", ""), ("GET", "/example"), ("DELETE", "/example"), ("GET", "/example/logs")],
)
def test_backend_401_includes_bearer_challenge(backend, method, suffix):
    http, custom, *_ = backend
    custom.get_namespaced_custom_object.side_effect = ApiException(status=401)
    custom.list_namespaced_custom_object.side_effect = ApiException(status=401)
    response = http.request(method, BASE + suffix, headers=HEADERS)
    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"
    assert response.json()["code"] == "unauthorized"


@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"])
def test_missing_ray_submission_is_not_retryable(backend, monkeypatch, follow, status):
    from ray.dashboard.modules import dashboard_sdk

    http, _, _, _, record, *_ = backend
    record["status"]["jobStatus"] = status
    sdk = object.__new__(logs.BoundedJobClient)
    sdk._address = "https://ray.example.com"
    sdk._cookies = None
    sdk._headers = HEADERS
    sdk._verify = True
    monkeypatch.setattr(jobs, "CodeFlareLogClient", Mock(return_value=sdk))
    monkeypatch.setattr(
        dashboard_sdk.requests,
        "request",
        Mock(return_value=Mock(status_code=404, text="sensitive-backend-body")),
    )
    response = http.get(BASE + "/example/logs", headers=HEADERS, params={"follow": follow})
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert "job 'example' in project 'team-a'" in response.json()["message"]
    assert "submission" in response.json()["message"].lower()
    assert "retry later" not in response.json()["message"]
    assert "sensitive-backend-body" not in response.text
    assert "text/event-stream" not in response.headers.get("content-type", "")


@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize("cluster_exists", [False, True])
def test_provisioning_job_id_is_not_proof_of_ray_submission(backend, follow, cluster_exists):
    http, custom, _, _, record, _, _, factory = backend
    # KubeRay allocates jobId and rayClusterName before submitting to Ray.
    record["status"]["jobStatus"] = ""
    record["status"]["jobDeploymentStatus"] = "Initializing"
    if not cluster_exists:
        custom.get_namespaced_custom_object.side_effect = [record, ApiException(status=404)]
    response = http.get(BASE + "/example/logs", headers=HEADERS, params={"follow": follow})
    assert response.status_code == 409
    assert response.json()["code"] == "logs_not_ready"
    assert custom.get_namespaced_custom_object.call_count == 1
    factory.assert_not_called()


@pytest.mark.parametrize("field", ["jobId", "rayClusterName"])
@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "STOPPED"])
def test_terminal_job_without_submission_metadata_is_not_retryable(backend, field, status):
    http, _, _, _, record, _, _, factory = backend
    record["status"]["jobStatus"] = status
    record["status"].pop(field)
    response = http.get(BASE + "/example/logs", headers=HEADERS)
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert "job 'example' in project 'team-a'" in response.json()["message"]
    factory.assert_not_called()


def test_delete_race_returns_conflict_without_retry(backend):
    http, custom, _, _, record, *_ = backend
    record["status"]["jobStatus"] = "SUCCEEDED"
    custom.delete_namespaced_custom_object.side_effect = ApiException(status=409)
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 409
    assert response.json()["code"] == "conflict"
    assert "Refresh" in response.json()["message"]
    assert custom.delete_namespaced_custom_object.call_count == 1


def test_log_snapshot_uses_scoped_gateway_and_caller_token(backend):
    http, _, _, sdk, _, _, _, factory = backend
    response = http.get(BASE + "/example/logs?tail_lines=2&follow=false", headers=HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "second\nthird\n"
    assert factory.call_args.args == ("https://ray.example.com/ray/team-a/example-cluster",)
    assert factory.call_args.kwargs == {"headers": HEADERS, "verify": True}
    sdk.get_job_logs.assert_called_once_with("raysubmit_123")


@pytest.mark.parametrize("configured, expected", [(None, 300), ("1", 1), ("7200", 7200)])
def test_follow_uses_sse_and_closes_sdk_iterator(backend, monkeypatch, configured, expected):
    http, _, _, sdk, *_ = backend
    setting = "TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS"
    if configured is None:
        monkeypatch.delenv(setting, raising=False)
    else:
        monkeypatch.setenv(setting, configured)
    http = TestClient(create_app(Authenticator(), Authorizer()))
    # Settings are captured when the application is created, not during streaming.
    monkeypatch.setenv(setting, "invalid-after-startup")
    timeout = Mock(wraps=asyncio.timeout)
    monkeypatch.setattr(logs.asyncio, "timeout", timeout)
    closed = []

    async def chunks(submission_id):
        try:
            yield "one\ntwo\n"
        finally:
            closed.append(True)

    sdk.tail_job_logs = chunks
    response = http.get(BASE + "/example/logs?follow=true", headers=HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: log\ndata: {"message": "one\\ntwo\\n"}\n\n' == response.text
    assert closed == [True]
    timeout.assert_called_once_with(expected)
    sdk.get_job_status.assert_called_once_with("raysubmit_123")
    sdk.get_job_logs.assert_not_called()


def test_follow_expiry_closes_sdk_iterator_and_requests_reconnection(backend, monkeypatch):
    http, _, _, sdk, *_ = backend
    monkeypatch.setenv("TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS", "7200")
    http = TestClient(create_app(Authenticator(), Authorizer()))
    real_timeout = asyncio.timeout
    timeout = Mock(side_effect=lambda seconds: real_timeout(0.01))
    monkeypatch.setattr(logs.asyncio, "timeout", timeout)
    closed = []

    async def chunks(submission_id):
        try:
            yield "ready"
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    sdk.tail_job_logs = chunks
    response = http.get(BASE + "/example/logs?follow=true", headers=HEADERS)
    assert response.status_code == 200
    assert 'event: log\ndata: {"message": "ready"}\n\n' in response.text
    assert response.text.endswith(
        'event: error\ndata: {"code":"stream_expired","message":"Reconnect to continue."}\n\n'
    )
    assert closed == [True]
    timeout.assert_called_once_with(7200)
    sdk.stop_job.assert_not_called()


@pytest.mark.parametrize("value", ["0", "-1", "", "invalid", "1.5", "nan", "inf"])
def test_invalid_log_stream_timeout_is_rejected_at_startup(monkeypatch, value):
    monkeypatch.setenv("TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS"):
        create_app(Authenticator(), Authorizer())


def test_follow_failure_is_sanitized_after_headers(backend):
    http, _, _, sdk, *_ = backend

    async def chunks(submission_id):
        yield "ready"
        raise RuntimeError("sensitive-stream-error")

    sdk.tail_job_logs = chunks
    response = http.get(BASE + "/example/logs?follow=true", headers=HEADERS)
    assert "event: error" in response.text
    assert "sensitive-stream-error" not in response.text


@pytest.mark.parametrize(
    "case, expected",
    [("pending", 409), ("gone", 410), ("unowned", 403), ("forbidden", 403), ("unavailable", 503)],
)
def test_logs_fail_before_streaming_when_backend_unavailable(backend, case, expected):
    http, custom, _, sdk, record, cluster, *_ = backend
    if case == "pending":
        record["status"].pop("jobId")
        record["status"]["jobStatus"] = "PENDING"
    elif case == "gone":
        custom.get_namespaced_custom_object.side_effect = [record, ApiException(status=404)]
    elif case == "unowned":
        cluster["metadata"]["ownerReferences"] = []
    elif case == "forbidden":
        sdk.get_job_status.side_effect = ApiError(403, "forbidden", "Denied.")
    else:
        sdk.get_job_status.side_effect = RuntimeError("sensitive-error")
    response = http.get(BASE + "/example/logs?follow=true", headers=HEADERS)
    assert response.status_code == expected
    assert "sensitive-error" not in response.text


@pytest.mark.parametrize(
    "value",
    [
        "http://untrusted.example",
        "https://user:password@ray.example.com",
        "https://ray.example.com/other",
        "",
    ],
)
def test_invalid_gateway_never_receives_credentials(backend, monkeypatch, value):
    http, _, _, _, _, _, _, factory = backend
    monkeypatch.setenv("TRAINING_SERVICE_RAY_GATEWAY_URL", value)
    assert http.get(BASE + "/example/logs", headers=HEADERS).status_code == 503
    factory.assert_not_called()


@pytest.mark.parametrize(
    "query", ["limit=0", "limit=oops", "status=ALL", "page_token=invalid!", "page_token=W10="]
)
def test_invalid_list_queries_return_contract_400(backend, query):
    assert backend[0].get(BASE + "?" + query, headers=HEADERS).status_code == 400


def test_missing_project_is_not_reported_as_empty(backend):
    http, custom, core, *_ = backend
    core.read_namespace.side_effect = ApiException(status=404)
    response = http.get(BASE, headers=HEADERS)
    assert response.status_code == 404
    assert response.json() == {
        "code": "not_found",
        "message": "Project 'team-a' was not found.",
    }
    custom.list_namespaced_custom_object.assert_not_called()


def test_missing_job_api_does_not_claim_existing_project_is_missing(backend):
    http, custom, core, *_ = backend
    custom.list_namespaced_custom_object.side_effect = ApiException(status=404)
    response = http.get(BASE, headers=HEADERS)
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert "Project 'team-a' was not found" not in response.json()["message"]
    core.read_namespace.assert_called_once()


@pytest.mark.parametrize(
    "method, suffix",
    [("get", ""), ("get", "/example"), ("get", "/example/logs"), ("delete", "/example")],
)
def test_denied_project_cannot_reach_backends(backend, method, suffix):
    http, custom, _, _, _, _, authorizer, factory = backend
    authorizer.allowed = False
    assert http.request(method, BASE + suffix, headers=HEADERS).status_code == 403
    assert not custom.mock_calls
    factory.assert_not_called()


def test_caller_clients_do_not_share_credentials(backend, monkeypatch):
    http = backend[0]
    original = kubernetes_client.client.ApiClient
    configurations = []

    def capture(configuration):
        configurations.append(configuration)
        return original(configuration)

    monkeypatch.setattr(kubernetes_client.client, "ApiClient", capture)

    def read(token):
        return http.get(BASE + "/example", headers={"Authorization": f"Bearer {token}"}).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(read, ["caller-a", "caller-b"])) == [200, 200]
    assert {c.api_key["authorization"] for c in configurations} == {
        "Bearer caller-a",
        "Bearer caller-b",
    }
    assert len({id(c) for c in configurations}) == 2
    assert all(c.verify_ssl and c.retries == 0 for c in configurations)


def test_pending_jobs_ignore_spurious_completion_time():
    assert "finished_at" not in jobs.detail(job(), "team-a")
    record = deepcopy(job(status="FAILED"))
    record["status"]["message"] = "Training failed."
    assert jobs.detail(record, "team-a")["failure_reason"] == "Training failed."


def test_runtime_resources_do_not_reconstruct_submission(backend):
    http, _, _, _, record, *_ = backend
    record["spec"]["rayClusterSpec"] = {
        "workerGroupSpecs": [
            {
                "replicas": 2,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "resources": {
                                    "requests": {
                                        "cpu": "8",
                                        "memory": "32Gi",
                                        "nvidia.com/gpu": "1",
                                    }
                                }
                            }
                        ]
                    }
                },
            }
        ]
    }
    data = http.get(BASE + "/example", headers=HEADERS).json()
    assert data["resources"] == {
        "num_nodes": 2,
        "gpus_per_node": 1,
        "gpu_type": "nvidia.com/gpu",
        "cpus_per_node": "8",
        "memory_per_node": "32Gi",
    }
    assert "resources" not in data["submission"]


def test_heterogeneous_head_and_workers_are_not_reported_as_one_shape():
    def group(cpu):
        return {"template": {"spec": {"containers": [{"resources": {"requests": {"cpu": cpu}}}]}}}

    cluster = {
        "headGroupSpec": group("2"),
        "workerGroupSpecs": [dict(group("16"), replicas=2)],
    }
    assert jobs._resources(cluster) is None


def test_swagger_does_not_advertise_scaffold_responses_for_implemented_operations(backend):
    document = backend[0].get("/openapi.json").json()
    assert "501" not in document["paths"]["/projects/{project}/training-jobs"]["get"]["responses"]
    assert document["paths"]["/projects/{project}/training-jobs"]["post"]["responses"]["501"] == {
        "$ref": "#/components/responses/501NotImplemented"
    }


def test_sdk_environment_override_is_rejected_before_credentials_are_sent(backend, monkeypatch):
    http, _, _, _, _, _, _, factory = backend
    monkeypatch.setenv("RAY_API_SERVER_ADDRESS", "https://unrelated.example.com")
    assert http.get(BASE + "/example/logs", headers=HEADERS).status_code == 503
    factory.assert_not_called()


@pytest.mark.parametrize("query", ["tail_lines=0", "follow=invalid"])
def test_invalid_log_queries_return_contract_400(backend, query):
    assert backend[0].get(BASE + "/example/logs?" + query, headers=HEADERS).status_code == 400


@pytest.mark.parametrize(
    "status, expected", [(200, 200), (302, 503), (401, 401), (403, 403), (404, 404), (500, 503)]
)
def test_real_sdk_transport_bounds_timeouts_and_sanitizes_errors(monkeypatch, status, expected):
    from ray.dashboard.modules import dashboard_sdk

    response = Mock(status_code=status, text="sensitive-gateway-response")

    def request(*args, **kwargs):
        kwargs["hooks"]["response"](response)
        return response

    transport = Mock(side_effect=request)
    monkeypatch.setattr(dashboard_sdk.requests, "request", transport)
    sdk = object.__new__(logs.BoundedJobClient)
    sdk._address = "https://ray.example.com/ray/team-a/example-cluster"
    sdk._cookies = None
    sdk._headers = HEADERS
    sdk._verify = True
    if expected == 200:
        assert sdk._do_request("GET", "/api/jobs/raysubmit_123/logs") is response
    else:
        with pytest.raises(ApiError) as error:
            sdk._do_request("GET", "/api/jobs/raysubmit_123/logs")
        assert error.value.status_code == expected
        assert "sensitive-gateway-response" not in str(error.value)
    assert transport.call_args.kwargs["timeout"] == (5, 15)
    assert transport.call_args.kwargs["allow_redirects"] is False
    assert transport.call_args.kwargs["verify"] is True


def test_handlers_use_injected_backend_without_ray_resource_assumptions():
    backend = Mock(spec=jobs.JobsBackend)
    backend.backend_name = "ray"  # The current public contract still supports Ray only.
    backend.api_group = "trainer.kubeflow.org"
    backend.resource = "trainjobs"
    backend.list_jobs.return_value = {"items": [], "total_count": 0}
    backend.get_job.return_value = jobs.detail(job(), "team-a")
    backend.delete_job.return_value = None
    log_client = Mock()
    log_client.get_job_logs.return_value = "backend-neutral log\n"
    backend.log_client.return_value = (log_client, "opaque-job-id")
    authorizer = Authorizer()
    http = TestClient(create_app(Authenticator(), authorizer, backend=backend))

    assert http.get(BASE, headers=HEADERS).json() == {"items": [], "total_count": 0}
    assert http.get(BASE + "/example", headers=HEADERS).status_code == 200
    assert http.delete(BASE + "/example", headers=HEADERS).status_code == 204
    response = http.get(BASE + "/example/logs", headers=HEADERS)
    assert response.status_code == 200
    assert response.text == "backend-neutral log\n"
    backend.list_jobs.assert_called_once()
    backend.get_job.assert_called_once()
    backend.delete_job.assert_called_once()
    backend.log_client.assert_called_once()
    log_client.get_job_logs.assert_called_once_with("opaque-job-id")
    assert [permission.verb for _, permission in authorizer.calls] == [
        "list",
        "get",
        "delete",
        "get",
    ]
    assert all(
        permission.group == "trainer.kubeflow.org" and permission.resource == "trainjobs"
        for _, permission in authorizer.calls
    )
