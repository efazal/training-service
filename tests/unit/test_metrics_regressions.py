"""Failure and concurrency regressions for training telemetry."""

import json
import sys
import threading
import time
from http.client import IncompleteRead
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.unit.test_metrics import (
    CONFIG,
    RUN,
    FakeTracking,
    bound_response,
    context,
    sample,
    tracked_job,
)

from training_service import ray_entrypoint
from training_service.errors import ApiError
from training_service.metrics import client, runtime, tracking
from training_service_api.models.create_job_request import CreateJobRequest
from training_service_api.models.training_configuration import TrainingConfiguration


@pytest.mark.parametrize("model", [CreateJobRequest, TrainingConfiguration])
@pytest.mark.parametrize("name", [" experiment", "\nexperiment", "experiment "])
def test_experiment_name_accepts_nonwhitespace_anywhere(model, name):
    assert (
        model.model_validate({**sample(), "mlflow_experiment_name": name}).mlflow_experiment_name
        == name
    )


@pytest.mark.parametrize("uri", ["https://tracking.example/?", "https://tracking.example/#"])
def test_empty_url_delimiters_rejected(uri):
    assert not client.valid_url(uri)


def test_truncated_response_is_a_sanitized_tracking_failure():
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.side_effect = IncompleteRead(b"withheld", 1)
    obj._opener.open = Mock(return_value=response)
    with pytest.raises(client.TrackingError) as raised:
        obj.request("POST", "runs/log-batch", {})
    assert "withheld" not in str(raised.value)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"experiment": None},
        {"experiment": {"experiment_id": None}},
        {"experiment": {"experiment_id": "not-an-id"}},
    ],
)
def test_malformed_experiment_responses_are_tracking_errors(response):
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    obj.request = Mock(return_value=response)
    with pytest.raises(client.TrackingError):
        obj.experiment("Default")


@pytest.mark.parametrize("response", [{}, {"run": None}, {"run": {"info": {}}}])
def test_malformed_run_responses_are_tracking_errors(response):
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    obj.request = Mock(return_value=response)
    with pytest.raises(client.TrackingError):
        obj.create_run("42", "job", {})
    with pytest.raises(client.TrackingError):
        obj.get_run(RUN)


def test_deleted_default_has_actionable_submission_error(monkeypatch):
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    obj.request = Mock(
        return_value={"experiment": {"experiment_id": "42", "lifecycle_stage": "deleted"}}
    )
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    with pytest.raises(ApiError, match="Restore it in MLflow") as raised:
        tracking.reserve(
            CONFIG["tracking_uri"], "team-alpha", "caller", None, "job", "sft", "user", {}
        )
    assert raised.value.status_code == 409
    assert obj.request.call_count == 1


@pytest.mark.parametrize(
    "status",
    [
        None,
        SimpleNamespace(allowed=None, evaluation_error=""),
        SimpleNamespace(allowed=False, evaluation_error="authorizer unavailable"),
    ],
)
def test_indeterminate_authorization_is_unavailable(monkeypatch, status):
    monkeypatch.setattr(tracking.config, "load_incluster_config", lambda **kwargs: None)
    auth = Mock()
    auth.create_subject_access_review.return_value.status = status
    monkeypatch.setattr(tracking.client, "AuthorizationV1Api", lambda api: auth)
    with pytest.raises(ApiError) as raised:
        tracking.verify_training_access("team-alpha")
    assert raised.value.status_code == 503


def test_failed_submission_deletes_the_reserved_run():
    obj = Mock()
    tracking.TrackedRun(obj, RUN, "42", "Default").failed_submission()
    obj.delete_run.assert_called_once_with(RUN)
    obj.finish.assert_not_called()


def test_failed_submission_marks_failed_when_deletion_fails():
    obj = Mock()
    obj.delete_run.side_effect = client.TrackingError()
    obj.log.side_effect = client.TrackingError()
    tracking.TrackedRun(obj, RUN, "42", "Default").failed_submission()
    obj.finish.assert_called_once_with(RUN, "FAILED")


@pytest.mark.parametrize("change", ["uid", "name", "experiment"])
def test_copied_or_mismatched_annotations_never_finalize_another_run(monkeypatch, change):
    obj = Mock()
    obj.get_run.return_value = bound_response()
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    job = tracked_job()
    if change == "experiment":
        obj.get_run.return_value["info"]["experiment_id"] = "43"
    else:
        job["metadata"][change] = "copied-job"
    tracking.finalize_deleted(job, "team-alpha", "caller", None)
    assert tracking.public_metadata(job, "team-alpha", "caller", None) == {}
    obj.finish.assert_not_called()


def test_delete_finalizes_active_driver_without_relying_on_sigterm(monkeypatch):
    obj = Mock()
    obj.get_run.return_value = bound_response(**{"job.started_at": "2026-11-20T12:00:00Z"})
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    job = tracked_job()
    job["status"] = {"jobStatus": "RUNNING"}
    tracking.finalize_deleted(job, "team-alpha", "caller", None)
    obj.finish.assert_called_once_with(RUN, "KILLED")
    obj.log.assert_called_once()


def test_terminal_update_survives_summary_failures(monkeypatch):
    obj = FakeTracking()
    obj.log = Mock(side_effect=client.TrackingError())
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: obj)
    record = runtime.RuntimeRecorder(CONFIG)
    record.close("FINISHED")
    assert obj.finished == [(RUN, "FINISHED")]


def test_batches_preserve_every_loss_step_and_event_timestamp(monkeypatch):
    ready, release = threading.Event(), threading.Event()
    obj = FakeTracking()
    original = obj.log_batch

    def delayed(run_id, metrics, tags):
        ready.set()
        assert release.wait(2)
        time.sleep(0.005)
        original(run_id, metrics, tags)

    obj.log_batch = delayed
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: obj)
    record = runtime.RuntimeRecorder(CONFIG)
    assert ready.wait(2)
    clock = {"now": 100.0}
    monkeypatch.setattr(runtime.time, "time", lambda: clock["now"])
    try:
        for step in range(100):
            clock["now"] = 100.0 + step
            record.event("on_log", context(step=step, loss=1 / (step + 1)))
    finally:
        clock["now"] = 10000.0
        release.set()
        record.close("FINISHED")
    losses = [point for batch in obj.batches for point in batch if point["key"] == "train/loss"]
    assert [point["step"] for point in losses] == list(range(100))
    assert [point["timestamp"] for point in losses] == [(100 + step) * 1000 for step in range(100)]
    assert len(obj.batches) < 20


def test_drain_deadline_counts_inflight_and_queued_events(monkeypatch):
    ready, release = threading.Event(), threading.Event()
    obj = FakeTracking()
    original = obj.log_batch

    def blocked(*args):
        ready.set()
        assert release.wait(2)
        original(*args)

    obj.log_batch = blocked
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: obj)
    record = runtime.RuntimeRecorder(CONFIG)
    assert ready.wait(2)
    record.emit({"loss": 0.5})
    monkeypatch.setattr(record, "DRAIN_TIMEOUT", 0.01)
    try:
        record.close("FINISHED")
        summaries = [
            call for _, call in obj.logs if "tracking/unresolved_events" in call.get("metrics", {})
        ]
        assert summaries[-1]["metrics"]["tracking/unresolved_events"] == 2
        assert summaries[-1]["tags"]["tracking.delivery"] == "incomplete"
    finally:
        release.set()
        record._writer.join(2)


def test_driver_preserves_rank_delivery_failure(monkeypatch):
    obj = FakeTracking()
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: obj)
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS", "another-process:1")
    original = obj.log_batch
    obj.log_batch = Mock(side_effect=client.TrackingError())
    rank = runtime.RuntimeRecorder(CONFIG)
    rank._queue.join()
    obj.log_batch = original
    rank.close()
    monkeypatch.delenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS")
    driver = runtime.RuntimeRecorder(CONFIG)
    driver.close("FINISHED")
    summary = obj.logs[-1][1]
    assert summary["tags"]["tracking.delivery"] == "incomplete"
    assert summary["metrics"]["tracking/undelivered_events"] > 0


def test_missing_expected_rank_prevents_complete_delivery(monkeypatch):
    obj = FakeTracking()
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: obj)
    record = runtime.RuntimeRecorder({**CONFIG, "expected_ranks": "2"})
    record.emit(tags={"tracking.unified_callbacks": "true"})
    record.event("on_log", context())
    record.close("FINISHED")
    assert obj.logs[-1][1]["tags"]["tracking.delivery"] == "incomplete"


def test_nonfinite_loss_is_visible_without_poisoning_json(monkeypatch):
    obj = FakeTracking()
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: obj)
    record = runtime.RuntimeRecorder(CONFIG)
    record.event("on_log", context(loss=float("nan"), metrics={"accuracy": float("inf")}))
    record.close("FINISHED")
    json.dumps(obj.batches, allow_nan=False)
    tags = obj.get_run(RUN)["data"]["tags"]
    assert {t["key"]: t["value"] for t in tags}["training.nonfinite_metrics"] == "true"
    assert obj.logs[-1][1]["metrics"]["tracking/nonfinite_values"] == 2


def test_rotated_checkpoint_is_omitted_not_reported_as_zero(monkeypatch, tmp_path):
    ready, release = threading.Event(), threading.Event()
    obj = FakeTracking()
    original = obj.log_batch

    def blocked(*args):
        ready.set()
        assert release.wait(2)
        original(*args)

    obj.log_batch = blocked
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: obj)
    record = runtime.RuntimeRecorder(CONFIG)
    assert ready.wait(2)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    record.set_artifact_root(str(tmp_path))
    (checkpoint / "weights").write_bytes(b"abc")
    record.event("on_save", context(metrics={"checkpoint_path": str(checkpoint)}))
    (checkpoint / "weights").unlink()
    checkpoint.rmdir()
    release.set()
    record.close("FINISHED")
    assert not any(
        point["key"] == "checkpoint/size_bytes" for batch in obj.batches for point in batch
    )


def test_training_results_recorded_before_failed_upload(monkeypatch, tmp_path):
    record = Mock(configuration=dict(CONFIG))
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_RUN", json.dumps(CONFIG))
    monkeypatch.setitem(
        sys.modules,
        "training_service_metrics_runtime",
        SimpleNamespace(current_recorder=lambda: record),
    )
    monkeypatch.setattr(ray_entrypoint, "_ensure_s3_runtime", lambda *args: None)
    monkeypatch.setattr(ray_entrypoint, "_input", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(ray_entrypoint, "_run_training", lambda *args: {"loss": 0.5})
    monkeypatch.setattr(ray_entrypoint, "_upload_s3", Mock(side_effect=OSError("upload failed")))
    with pytest.raises(OSError, match="upload failed"):
        ray_entrypoint._execute(
            {
                "algorithm": "lora_sft",
                "parameters": {},
                "model": {"scheme": "pvc"},
                "dataset": {"scheme": "pvc"},
                "output": {"scheme": "s3"},
                "gpus_per_node": 1,
                "num_workers": 0,
                "mlflow": CONFIG,
            }
        )
    assert record.emit.call_args_list[0].args == ({"result": {"loss": 0.5}},)


def test_batch_entity_limit_includes_tags():
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    obj.request = Mock(return_value={})
    metrics = [{"key": "train/loss", "value": 0.5, "timestamp": i, "step": i} for i in range(1000)]
    obj.log_batch(RUN, metrics, {"training.phase": "training"})
    bodies = [call.args[2] for call in obj.request.call_args_list]
    assert all(sum(len(body[k]) for k in ("metrics", "tags", "params")) <= 1000 for body in bodies)
    assert [point for body in bodies for point in body["metrics"]] == metrics
    assert sum(len(body["tags"]) for body in bodies) == 1


@pytest.mark.parametrize("status", [[], {}, None, 42])
def test_malformed_status_is_sanitized(status):
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "test-only")
    obj.request = Mock(
        return_value={"run": {"info": {"run_id": RUN, "experiment_id": "42", "status": status}}}
    )
    with pytest.raises(client.TrackingError):
        obj.get_run(RUN)


def test_job_status_survives_gateway_trust_lookup_failure(monkeypatch):
    from contextlib import nullcontext

    from fastapi import Request

    from training_service import jobs

    backend = jobs.CodeFlareJobBackend()
    monkeypatch.setattr(jobs, "caller_client", lambda *args: nullcontext(object()))
    monkeypatch.setattr(jobs, "read_job", lambda *args: tracked_job())
    monkeypatch.setattr(
        backend,
        "metrics_ca_file",
        Mock(side_effect=ApiError(503, "metrics_unavailable", "Trust unavailable")),
    )
    request = Request({"type": "http", "headers": [(b"authorization", b"Bearer caller")]})
    request.state.caller_bearer_token = "caller"
    detail = backend.get_job(request, "team-alpha", "job")
    assert detail["id"] == "job"
    assert "mlflow_url" not in detail


def test_deleted_experiment_error_serializes_through_http_handler():
    from fastapi.testclient import TestClient
    from tests.unit.test_submission import FakeAuthenticator, FakeAuthorizer

    from training_service.main import create_app

    class Submitter:
        async def submit(self, *args):
            raise ApiError(
                409, "mlflow_experiment_deleted", "Restore the deleted experiment in MLflow."
            )

    response = TestClient(create_app(FakeAuthenticator(), FakeAuthorizer(), Submitter())).post(
        "/api/v1/projects/team-alpha/training-jobs",
        json=sample(),
        headers={"Authorization": "Bearer caller-token"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "mlflow_experiment_deleted"


def test_sigterm_flushes_cancelled_run_in_real_subprocess():
    import subprocess

    script = """
import os, signal, sys
from types import SimpleNamespace
from training_service import ray_entrypoint
record = SimpleNamespace(emit=lambda *a, **kw: None, start_sampling=lambda: None,
                         close=lambda status: print('CANCELLED:' + status, flush=True))
sys.modules['training_service_metrics_runtime'] = SimpleNamespace(current_recorder=lambda: record)
ray_entrypoint._execute = lambda cfg: os.kill(os.getpid(), signal.SIGTERM)
ray_entrypoint.main({'mlflow': {'run_id': 'synthetic'}})
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 143
    assert "CANCELLED:KILLED" in result.stdout
