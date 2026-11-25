"""Deleting MLflow history must not strand its RayJob behind a finalizer."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from tests.unit.test_job_operations import BASE, HEADERS
from tests.unit.test_job_operations import backend as backend
from tests.unit.test_metrics import CONFIG, RUN

from training_service import jobs
from training_service.metrics import client, tracking


@pytest.fixture
def deletion(backend, monkeypatch):
    http, custom, _, _, record, *_ = backend
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-a", lambda: "test-only")
    annotation = json.loads(tracking.TrackedRun(obj, RUN, "42", "Default").annotation())
    annotation["submission_id"] = "submission-uid"
    record["metadata"]["annotations"][tracking.ANNOTATION] = json.dumps(annotation)
    record["metadata"]["finalizers"] = ["example.com/other-finalizer"]
    run = {
        "info": {
            "run_id": RUN,
            "experiment_id": "42",
            "status": "RUNNING",
            "lifecycle_stage": "active",
        },
        "data": {
            "params": [{"key": "training.rayjob_uid", "value": record["metadata"]["uid"]}],
            "tags": [
                {"key": "training.project", "value": "team-a"},
                {"key": "training.job_id", "value": "example"},
            ],
        },
    }
    state = {"lifecycle": "active", "delete_on": None, "read_error": None, "update_error": None}

    def request(method, endpoint, data):
        if endpoint == "runs/get":
            if state["read_error"] is not None:
                raise client.TrackingError(state["read_error"])
            if state["lifecycle"] == "missing":
                raise client.TrackingError(404)
            result = deepcopy(run)
            result["info"]["lifecycle_stage"] = state["lifecycle"]
            return {"run": result}
        phase = "update" if endpoint == "runs/update" else "claim" if data["params"] else "tag"
        if state["delete_on"] == phase:
            state["lifecycle"] = state["race_lifecycle"]
        if state["lifecycle"] != "active":
            raise client.TrackingError(404 if state["lifecycle"] == "missing" else 400)
        if endpoint == "runs/log-batch":
            run["data"]["params"].extend(data["params"])
            return {}
        if endpoint == "runs/update":
            if state["update_error"] is not None:
                raise client.TrackingError(state["update_error"])
            run["info"]["status"] = data["status"]
            return {}
        raise AssertionError((method, endpoint))

    obj.request = Mock(side_effect=request)
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    monkeypatch.setattr(jobs.CodeFlareJobBackend, "metrics_ca_file", lambda self: None)
    marker = Mock(submission_id="submission-uid", data={"run_id": RUN})
    monkeypatch.setattr(jobs.metrics_reservations, "load", Mock(return_value=marker))

    def patch(*args, **kwargs):
        metadata = kwargs["body"]["metadata"]
        assert metadata["uid"] == record["metadata"]["uid"]
        assert metadata["resourceVersion"] == record["metadata"]["resourceVersion"]
        record["metadata"].update(metadata)
        record["metadata"]["resourceVersion"] = str(int(metadata["resourceVersion"]) + 1)
        return deepcopy(record)

    def delete(**kwargs):
        assert not obj.request.called
        assert tracking.DELETION_FINALIZER in record["metadata"]["finalizers"]
        record["metadata"]["deletionTimestamp"] = "2026-11-24T09:11:32Z"
        record["metadata"]["finalizers"].append("foregroundDeletion")

    custom.patch_namespaced_custom_object.side_effect = patch
    custom.delete_namespaced_custom_object.side_effect = delete
    return http, custom, record, obj, marker, state


@pytest.mark.parametrize("lifecycle", ["missing", "deleted"])
@pytest.mark.parametrize("terminating", [False, True])
def test_delete_releases_finalizer_when_mlflow_run_is_already_gone(
    deletion, lifecycle, terminating
):
    http, custom, record, obj, marker, state = deletion
    state["lifecycle"] = lifecycle
    if terminating:
        record["metadata"].update(deletionTimestamp="2026-11-24T09:11:32Z")
        record["metadata"]["finalizers"].extend([tracking.DELETION_FINALIZER, "foregroundDeletion"])
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 204, response.text
    assert record["metadata"]["finalizers"] == ["example.com/other-finalizer", "foregroundDeletion"]
    assert custom.delete_namespaced_custom_object.call_count == (0 if terminating else 1)
    marker.complete.assert_called_once()
    assert marker.complete.call_args.args[0]["metadata"]["uid"] == record["metadata"]["uid"]
    assert all(call.args[0] == "GET" for call in obj.request.call_args_list)


@pytest.mark.parametrize("lifecycle", ["missing", "deleted"])
@pytest.mark.parametrize("phase", ["tag", "claim", "update"])
def test_delete_handles_run_removed_during_finalization(deletion, lifecycle, phase):
    http, _, record, _, marker, state = deletion
    state.update(delete_on=phase, race_lifecycle=lifecycle)
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 204, response.text
    assert record["metadata"]["finalizers"] == ["example.com/other-finalizer", "foregroundDeletion"]
    marker.complete.assert_called_once()
    assert marker.complete.call_args.args[0]["metadata"]["uid"] == record["metadata"]["uid"]


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_delete_keeps_retryable_binding_when_run_absence_is_not_confirmed(deletion, status):
    http, _, record, _, marker, state = deletion
    state["read_error"] = status
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 503
    assert response.json()["code"] == "metrics_unavailable"
    assert tracking.DELETION_FINALIZER in record["metadata"]["finalizers"]
    marker.complete.assert_not_called()


@pytest.mark.parametrize("status", [400, 404, 409])
def test_rejected_finalization_of_an_existing_run_still_requires_retry(deletion, status):
    http, _, record, _, marker, state = deletion
    state["update_error"] = status
    response = http.delete(BASE + "/example", headers=HEADERS)
    assert response.status_code == 503
    assert tracking.DELETION_FINALIZER in record["metadata"]["finalizers"]
    marker.complete.assert_not_called()
