"""Regression proof for the second MLflow review and lost-response recovery."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from kubernetes import client as kubernetes
from requests.certs import where
from tests.unit import test_submission as submission_tests
from tests.unit.test_job_operations import BASE, HEADERS
from tests.unit.test_job_operations import backend as backend
from tests.unit.test_metrics import CONFIG, RUN, bound_response, tracked_job

from training_service import jobs, log_config
from training_service.auth import CallerIdentity
from training_service.errors import ApiError
from training_service.metrics import client, reservation, tracking

PROJECT = "team-alpha"


@pytest.fixture
def server(monkeypatch):
    obj = client.TrackingClient(CONFIG["tracking_uri"], PROJECT, lambda: "caller")
    state = {"run": None, "creates": 0, "lost": False, "visible": True}

    def request(method, endpoint, body):
        if endpoint == "runs/create":
            state["creates"] += 1
            state["run"] = {
                "info": {"run_id": RUN, "experiment_id": "42", "status": "RUNNING"},
                "data": {"tags": deepcopy(body["tags"]), "params": []},
            }
            if state["lost"]:
                raise client.TrackingError()
            return {"run": deepcopy(state["run"])}
        if endpoint == "runs/search":
            assert method == "POST"
            assert body["experiment_ids"] == ["42"]
            assert body["filter"].startswith("tags.`training.submission_id` = '")
            return {"runs": [deepcopy(state["run"])] if state["visible"] and state["run"] else []}
        if endpoint == "runs/get":
            assert body == {"run_id": RUN}
            return {"run": deepcopy(state["run"])}
        if endpoint == "runs/log-batch":
            assert body["run_id"] == RUN
            existing = {x["key"]: x["value"] for x in state["run"]["data"]["params"]}
            for item in body["params"]:
                if item["key"] in existing and existing[item["key"]] != item["value"]:
                    raise client.TrackingError(400)
                existing[item["key"]] = item["value"]
            state["run"]["data"]["params"] = [
                {"key": key, "value": value} for key, value in existing.items()
            ]
            return {}
        raise AssertionError((method, endpoint))

    obj.request = Mock(side_effect=request)
    obj.experiment = Mock(return_value="42")
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    return obj, state


def reserve(marker):
    return tracking.reserve(
        CONFIG["tracking_uri"],
        PROJECT,
        "caller",
        None,
        "job",
        "sft",
        "alice",
        {},
        reservation=marker,
    )


def acquire(core):
    return reservation.acquire(core, PROJECT, "job", "digest", CONFIG["tracking_uri"], "Default")


def test_lost_mlflow_create_response_is_reconciled_without_another_create(server):
    obj, state = server
    core = submission_tests.FakeCoreApi()
    marker = acquire(core)
    state.update(lost=True, visible=False)
    with pytest.raises(ApiError) as raised:
        reserve(marker)
    assert raised.value.status_code == 503
    assert marker.data["phase"] == "creating"
    assert state["creates"] == 1
    # A new HTTP request can recover the now-visible create from the same marker.
    state["visible"] = True
    recovered = reserve(acquire(core))
    assert recovered.run_id == RUN
    assert state["creates"] == 1
    assert reserve(acquire(core)).run_id == RUN
    assert state["creates"] == 1
    assert obj.experiment.call_count == 1


def test_recover_late_kubernetes_create_by_exact_reservation(server, monkeypatch):
    obj, state = server
    core = submission_tests.FakeCoreApi()
    marker = acquire(core)
    run = reserve(marker)
    marker.update(submit_attempted="true")
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking.client, "CoreV1Api", lambda api: core)
    job = {
        "metadata": {
            "name": "job",
            "namespace": PROJECT,
            "uid": "late-created-job",
            "annotations": {tracking.ANNOTATION: run.annotation()},
        }
    }
    bound = tracking.bound_run(job, PROJECT, "caller", None)
    assert bound is not None
    assert {p["key"]: p["value"] for p in state["run"]["data"]["params"]}[
        "training.rayjob_uid"
    ] == "late-created-job"
    stored = next(iter(core.config_maps.values()))
    assert stored["metadata"]["ownerReferences"][0]["uid"] == "late-created-job"


def test_recovery_never_binds_a_replacement_for_an_already_captured_uid(server, monkeypatch):
    _, state = server
    core = submission_tests.FakeCoreApi()
    marker = acquire(core)
    run = reserve(marker)
    marker.update(submit_attempted="true")
    marker.bind_job({"metadata": {"name": "job", "uid": "original-created-uid"}})
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking.client, "CoreV1Api", lambda api: core)
    copied = {
        "metadata": {
            "name": "job",
            "uid": "replacement-uid",
            "annotations": {tracking.ANNOTATION: run.annotation()},
        }
    }
    assert tracking.bound_run(copied, PROJECT, "caller", None) is None
    assert "training.rayjob_uid" not in {p["key"] for p in state["run"]["data"]["params"]}


def test_transport_captures_its_own_create_response_before_name_lookup(monkeypatch):
    response = {"metadata": {"name": "job", "uid": "original-server-uid"}}
    monkeypatch.setattr(kubernetes.ApiClient, "call_api", lambda *args, **kwargs: response)
    with jobs._CallerApiClient() as api:
        assert (
            api.call_api(
                jobs._RAYJOB_CREATE_PATH,
                "POST",
                {"group": "ray.io", "version": "v1", "plural": "rayjobs"},
            )
            is response
        )
        assert api.created_rayjob["metadata"]["uid"] == "original-server-uid"


def test_submission_binds_create_response_even_if_the_name_lookup_would_return_replacement(
    server, monkeypatch
):
    _, state = server
    submission_tests._mock_sdk(monkeypatch)
    monkeypatch.setattr(jobs, "resolve_local_queue", lambda *args: None)
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "verify_training_access", lambda project: None)
    lookup = Mock(return_value={"metadata": {"uid": "replacement-uid"}})
    monkeypatch.setattr(jobs, "read_job", lookup)
    request = submission_tests._sample()
    jobs.CodeFlareJobBackend("quay.io/example/ray:2.58.0")._submit_sync(
        PROJECT, request, CallerIdentity("alice", None, ()), "caller"
    )
    lookup.assert_not_called()
    assert {p["key"]: p["value"] for p in state["run"]["data"]["params"]}[
        "training.rayjob_uid"
    ] == "created-rayjob-uid"


def test_metrics_ca_lookup_does_not_discover_ray_gateway(monkeypatch):
    monkeypatch.setenv("TRAINING_SERVICE_RAY_CA_DISCOVERY", "false")
    monkeypatch.delenv("TRAINING_SERVICE_RAY_GATEWAY_URL", raising=False)
    gateway = Mock(side_effect=ApiError(503, "logs_unavailable", "gateway not ready"))
    monkeypatch.setattr(log_config, "_discover_gateway", gateway)
    assert jobs.CodeFlareJobBackend().metrics_ca_file() == where()
    gateway.assert_not_called()


def test_existing_binding_can_recover_with_automatic_tracking_disabled(monkeypatch):
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_ENABLED", "false")
    monkeypatch.setattr(tracking.config, "load_incluster_config", lambda **kwargs: None)
    resource = {
        "metadata": {"generation": 1},
        "status": {
            "url": CONFIG["tracking_uri"],
            "conditions": [{"type": "Available", "status": "True", "observedGeneration": 1}],
        },
    }
    custom = Mock()
    custom.get_cluster_custom_object.return_value = resource
    monkeypatch.setattr(tracking.client, "CustomObjectsApi", lambda api: custom)
    obj = Mock()
    obj.get_run.return_value = bound_response()
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    assert tracking.discover() is None
    assert tracking.bound_run(tracked_job(), PROJECT, "caller", None) is not None


@pytest.mark.parametrize("deployment", ["Failed", "ValidationFailed"])
def test_infrastructure_failure_retains_failed_outcome(monkeypatch, deployment):
    obj = Mock()
    obj.get_run.return_value = bound_response()
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    job = tracked_job()
    job["status"] = {"jobDeploymentStatus": deployment}
    tracking.finalize_deleted(job, PROJECT, "caller", None)
    obj.finish.assert_called_once_with(RUN, "FAILED")


def test_maximum_supported_tracking_metadata_survives_annotation_roundtrip():
    uri = "https://tracking.example/" + "x" * 2020
    obj = client.TrackingClient(uri, PROJECT, lambda: "never-read")
    run = tracking.TrackedRun(obj, RUN, "42", "🧪" * 256)
    raw = run.annotation()
    assert len(raw) > 4096
    assert len(raw.encode()) <= tracking.ANNOTATION_LIMIT
    assert (
        tracking.metadata({"metadata": {"annotations": {tracking.ANNOTATION: raw}}}, PROJECT)[
            "run_id"
        ]
        == RUN
    )


def test_encoded_metric_batches_stay_below_rest_limit_and_keep_order():
    obj = client.TrackingClient(CONFIG["tracking_uri"], PROJECT, lambda: "caller")
    obj.request = Mock(return_value={})
    points = [{"key": "值" * 250, "value": 0.5, "timestamp": i, "step": i} for i in range(1200)]
    obj.log_batch(RUN, points, {"training.phase": "training"})
    bodies = [call.args[2] for call in obj.request.call_args_list]
    assert len(bodies) >= 2
    assert all(len(json.dumps(body).encode()) <= 1_000_000 for body in bodies)
    assert [point for body in bodies for point in body["metrics"]] == points
    assert all(not body["tags"] for body in bodies[:-1])
    assert bodies[-1]["tags"] == [{"key": "training.phase", "value": "training"}]


def test_mlflow_outage_does_not_prevent_authorized_workload_deletion(backend, monkeypatch):
    http, custom, _, _, record, *_ = backend
    monkeypatch.setattr(tracking, "metadata", lambda *args: {"run_id": RUN})
    monkeypatch.setattr(jobs.CodeFlareJobBackend, "metrics_ca_file", lambda self: None)
    bound = Mock(side_effect=client.TrackingError())
    monkeypatch.setattr(tracking, "bound_run", bound)
    events = []

    def patch(*args, **kwargs):
        record["metadata"].update(kwargs["body"]["metadata"])
        record["metadata"]["resourceVersion"] = str(int(record["metadata"]["resourceVersion"]) + 1)
        return deepcopy(record)

    def delete(**kwargs):
        events.append("delete")
        assert bound.call_count == 0
        record["metadata"]["deletionTimestamp"] = "2026-11-23T19:00:00Z"

    custom.patch_namespaced_custom_object.side_effect = patch
    custom.delete_namespaced_custom_object.side_effect = delete
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 503
    assert events == ["delete"]
    assert tracking.DELETION_FINALIZER in record["metadata"]["finalizers"]
