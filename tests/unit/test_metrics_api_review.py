"""Review regressions for durable, caller-scoped MLflow lifecycle updates."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Lock
from unittest.mock import Mock

import pytest
from kubernetes.client.exceptions import ApiException
from tests.unit import test_submission as submission_tests
from tests.unit.test_job_operations import BASE, HEADERS
from tests.unit.test_job_operations import backend as backend
from tests.unit.test_metrics import CONFIG, RUN, bound_response, tracked_job

from training_service import jobs
from training_service.auth import CallerIdentity
from training_service.errors import ApiError
from training_service.metrics import client, tracking


def test_active_delete_finalizes_even_when_driver_cannot_handle_sigterm(monkeypatch):
    obj = Mock()
    obj.get_run.return_value = bound_response(**{"job.started_at": "2026-11-23T10:00:00Z"})
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    job = tracked_job()
    job["status"] = {"jobStatus": "RUNNING"}
    tracking.finalize_deleted(job, "team-alpha", "caller", None)
    obj.finish.assert_called_once_with(RUN, "KILLED")


@pytest.mark.parametrize("status", ["FINISHED", "FAILED", "KILLED"])
def test_client_finish_preserves_an_existing_terminal_result(status):
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    obj.get_run = Mock(return_value=bound_response(status))
    obj.request = Mock()
    obj.finish(RUN, "KILLED")
    obj.request.assert_not_called()


def test_terminal_intent_survives_ambiguous_write_and_competing_result():
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    run = bound_response()
    winner = {"status": "FINISHED", "end_time": 123456}
    competing = deepcopy(run)
    competing["data"]["params"].append(
        {"key": "training.terminal_result", "value": json.dumps(winner)}
    )
    obj.get_run = Mock(side_effect=[run, competing])
    obj.log = Mock(side_effect=client.TrackingError(400))
    obj.request = Mock(return_value={})
    obj.finish(RUN, "KILLED")
    obj.request.assert_called_once_with("POST", "runs/update", {"run_id": RUN, **winner})


def test_unconfirmed_terminal_intent_does_not_write_a_conflicting_status():
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    obj.get_run = Mock(return_value=bound_response())
    obj.log = Mock(side_effect=client.TrackingError())
    obj.request = Mock(return_value={})
    with pytest.raises(client.TrackingError):
        obj.finish(RUN, "KILLED")
    obj.request.assert_not_called()


def test_concurrent_completion_and_cancellation_converge_on_one_result():
    lock, readers = Lock(), Barrier(2)
    server = {"result": None, "status": "RUNNING", "reads": 0, "writes": []}

    def get_run(run_id):
        with lock:
            run = bound_response(server["status"])
            if server["result"]:
                run["data"]["params"].append(
                    {"key": "training.terminal_result", "value": server["result"]}
                )
            server["reads"] += 1
            first_read = server["reads"] <= 2
        if first_read:
            readers.wait(timeout=3)
        return run

    def log(run_id, *, params):
        with lock:
            if server["result"] is not None:
                raise client.TrackingError(400)
            server["result"] = params["training.terminal_result"]

    def request(method, endpoint, data):
        assert (method, endpoint) == ("POST", "runs/update")
        with lock:
            server["writes"].append(data)
            server["status"] = data["status"]
        return {}

    def finish(status):
        obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
        obj.get_run, obj.log, obj.request = get_run, log, request
        obj.finish(RUN, status)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(finish, status) for status in ("FINISHED", "KILLED")]
        for future in futures:
            future.result(timeout=5)
    winner = json.loads(server["result"])
    assert server["writes"]
    assert all(write == {"run_id": RUN, **winner} for write in server["writes"])
    assert server["status"] == winner["status"]


def test_tracking_uses_verified_token_for_lowercase_bearer(backend, monkeypatch):
    http, _, _, _, record, *_ = backend
    monkeypatch.setattr(tracking, "metadata", lambda *args: {"run_id": RUN})
    public = Mock(return_value={})
    monkeypatch.setattr(tracking, "public_metadata", public)
    monkeypatch.setattr(jobs.CodeFlareJobBackend, "metrics_ca_file", lambda self: None)
    response = http.get(BASE + "/example", headers={"Authorization": "bearer unit-test-token"})
    assert response.status_code == 200
    assert public.call_args.args == (record, "team-a", "unit-test-token", None)


def test_tracking_finalization_failure_retains_binding_for_delete_retry(backend, monkeypatch):
    http, custom, _, _, record, *_ = backend
    monkeypatch.setattr(tracking, "metadata", lambda *args: {"run_id": RUN})
    monkeypatch.setattr(
        tracking, "bound_run", lambda *args: (Mock(), {"run_id": RUN}, bound_response())
    )
    finalize = Mock(side_effect=[client.TrackingError(), None])
    monkeypatch.setattr(tracking, "finalize_deleted", finalize)
    monkeypatch.setattr(jobs.CodeFlareJobBackend, "metrics_ca_file", lambda self: None)

    def patch(*args, **kwargs):
        metadata = kwargs["body"]["metadata"]
        assert metadata["resourceVersion"] == record["metadata"]["resourceVersion"]
        assert metadata["uid"] == record["metadata"]["uid"]
        record["metadata"].update(metadata)
        record["metadata"]["resourceVersion"] = str(int(metadata["resourceVersion"]) + 1)
        return deepcopy(record)

    def delete(**kwargs):
        assert tracking.DELETION_FINALIZER in record["metadata"]["finalizers"]
        record["metadata"]["deletionTimestamp"] = "2026-11-23T10:00:00Z"
        record["metadata"]["finalizers"].append("foregroundDeletion")

    custom.patch_namespaced_custom_object.side_effect = patch
    custom.delete_namespaced_custom_object.side_effect = delete
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 503
    assert response.json()["code"] == "metrics_unavailable"
    assert tracking.DELETION_FINALIZER in record["metadata"]["finalizers"]
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 204
    assert record["metadata"]["finalizers"] == ["foregroundDeletion"]
    assert custom.delete_namespaced_custom_object.call_count == 1
    assert finalize.call_count == 2
    assert finalize.call_args.args[2] == "unit-test-token"


@pytest.mark.parametrize("lookup", ["accepted", "absent", "unavailable"])
@pytest.mark.parametrize("failure", ["transport", "post_create"])
def test_ambiguous_submit_retains_reserved_run_even_when_first_lookup_is_absent(
    monkeypatch, lookup, failure
):
    core = submission_tests._mock_sdk(monkeypatch)
    monkeypatch.setattr(jobs, "resolve_local_queue", lambda *args: None)
    backend = jobs.CodeFlareJobBackend("quay.io/example/ray:2.58.0")
    monkeypatch.setattr(backend, "metrics_ca_file", lambda: None)
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "verify_training_access", lambda project: None)
    run = Mock(run_id=RUN)
    run.metadata.return_value = {"run_id": RUN}
    run.configuration.return_value = {"run_id": RUN}
    monkeypatch.setattr(tracking, "reserve", lambda *args, **kwargs: run)
    error = OSError("lost create response") if failure == "transport" else ApiException(status=403)
    submission_tests.FakeRayJob.error = error

    def accepted(*args):
        job = tracked_job()
        value = json.loads(job["metadata"]["annotations"][tracking.ANNOTATION])
        value["submission_id"] = next(iter(core.config_maps.values()))["metadata"]["uid"]
        job["metadata"]["annotations"][tracking.ANNOTATION] = json.dumps(value)
        return job

    read = Mock(side_effect=accepted)
    if lookup != "accepted":
        read.side_effect = ApiError(
            404 if lookup == "absent" else 503, "not_found", "lookup unavailable"
        )
    monkeypatch.setattr(jobs, "read_job", read)
    with pytest.raises(type(error)):
        backend._submit_sync(
            "team-alpha", submission_tests._sample(), CallerIdentity("alice", None, ()), "caller"
        )
    read.assert_called_once()
    # The original create can still commit after the timeout and first GET.
    run.failed_submission.assert_not_called()
    assert run.bind.called == (lookup == "accepted")


def test_delete_failure_keeps_tracking_finalizer_and_never_finalizes(backend, monkeypatch):
    http, custom, _, _, record, *_ = backend
    monkeypatch.setattr(tracking, "metadata", lambda *args: {"run_id": RUN})
    monkeypatch.setattr(
        tracking, "bound_run", lambda *args: (Mock(), {"run_id": RUN}, bound_response())
    )
    monkeypatch.setattr(jobs.CodeFlareJobBackend, "metrics_ca_file", lambda self: None)
    patched = deepcopy(record)
    patched["metadata"]["resourceVersion"] = "43"
    patched["metadata"]["finalizers"] = [tracking.DELETION_FINALIZER]
    custom.patch_namespaced_custom_object.return_value = patched
    custom.delete_namespaced_custom_object.side_effect = ApiException(status=409)
    finalize = Mock()
    monkeypatch.setattr(tracking, "finalize_deleted", finalize)
    assert http.delete(BASE + "/example", headers=HEADERS).status_code == 409
    finalize.assert_not_called()
    assert custom.patch_namespaced_custom_object.call_count == 1


def test_terminating_job_does_not_reinstall_a_released_tracking_finalizer(backend, monkeypatch):
    http, custom, _, _, record, *_ = backend
    record["metadata"].update(
        deletionTimestamp="2026-11-23T10:00:00Z", finalizers=["foregroundDeletion"]
    )
    monkeypatch.setattr(tracking, "metadata", lambda *args: {"run_id": RUN})
    monkeypatch.setattr(
        tracking, "bound_run", lambda *args: (Mock(), {"run_id": RUN}, bound_response("KILLED"))
    )
    monkeypatch.setattr(jobs.CodeFlareJobBackend, "metrics_ca_file", lambda self: None)
    finalize = Mock()
    monkeypatch.setattr(tracking, "finalize_deleted", finalize)
    custom.patch_namespaced_custom_object.side_effect = ApiException(status=422)
    assert http.delete(BASE + "/example", headers=HEADERS).status_code == 204
    custom.patch_namespaced_custom_object.assert_not_called()
    custom.delete_namespaced_custom_object.assert_not_called()
    finalize.assert_called_once()


def test_pending_tracking_finalization_ignores_a_reused_cluster_name(backend, monkeypatch):
    http, custom, _, _, record, cluster, *_ = backend
    record["metadata"].update(
        deletionTimestamp="2026-11-23T10:00:00Z",
        finalizers=[tracking.DELETION_FINALIZER, "foregroundDeletion"],
    )
    cluster["metadata"]["ownerReferences"][0]["uid"] = "unrelated-new-rayjob-uid"
    monkeypatch.setattr(tracking, "metadata", lambda *args: {"run_id": RUN})
    monkeypatch.setattr(
        tracking, "bound_run", lambda *args: (Mock(), {"run_id": RUN}, bound_response())
    )
    monkeypatch.setattr(jobs.CodeFlareJobBackend, "metrics_ca_file", lambda self: None)
    monkeypatch.setattr(tracking, "finalize_deleted", Mock())
    custom.patch_namespaced_custom_object.return_value = record
    assert http.delete(BASE + "/example", headers=HEADERS).status_code == 204
    assert all(
        call.args[3] == "rayjobs" for call in custom.get_namespaced_custom_object.call_args_list
    )
    assert custom.patch_namespaced_custom_object.call_args.kwargs["body"]["metadata"][
        "finalizers"
    ] == ["foregroundDeletion"]
    custom.delete_namespaced_custom_object.assert_not_called()
