"""Numeric integrity and bounded finalization regressions for the runtime."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from training_service.metrics import runtime

CONFIG = {"tracking_uri": "https://tracking.example", "project": "team", "run_id": "a" * 32}


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.delenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    client = Mock()
    client.get_run.return_value = {"data": {"tags": []}}
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(runtime.time, "sleep", Mock())
    record = runtime.RuntimeRecorder(CONFIG)
    record._queue.join()
    yield record
    record.close()
    record._writer.join(2)
    assert not record._writer.is_alive()


def context(**kwargs):
    return SimpleNamespace(
        **{
            "step": 3,
            "epoch": 0.5,
            "is_main_process": True,
            "loss": 0.25,
            "learning_rate": None,
            "metrics": {},
            **kwargs,
        }
    )


def points(record):
    return [point for args in record.client.log_batch.call_args_list for point in args.args[1]]


@pytest.mark.parametrize(
    "invalid",
    [10**400, -(10**400), float("inf"), float("-inf"), float("nan")],
    ids=["overflow", "negative-overflow", "inf", "negative-inf", "nan"],
)
def test_unrepresentable_numeric_values_are_accounted_without_losing_valid_loss(invalid):
    omitted = []
    values = runtime.numeric_metrics(
        {"backend": {"huge": invalid}, "train/loss": 0.25}, nonfinite=omitted
    )
    assert values == {"train/loss": 0.25}
    assert omitted == ["backend/huge"]
    assert runtime.numeric_metrics({"huge": invalid, "loss": 0.25}) == {"loss": 0.25}


def test_overflow_in_backend_metrics_keeps_training_loss_and_delivery_accounting(recorder):
    recorder.event("on_log", context(metrics={"huge": 10**400, "accuracy": 0.8}))
    recorder.close("FINISHED")
    values = {point["key"]: point["value"] for point in points(recorder)}
    assert values["train/loss"] == 0.25
    assert values["train/accuracy"] == 0.8
    assert "train/huge" not in values
    assert recorder.client.log.call_args.kwargs["metrics"]["tracking/nonfinite_values"] == 1
    assert recorder.client.log.call_args.kwargs["tags"]["training.nonfinite_metrics"] == "true"
    recorder.client.finish.assert_called_once_with(CONFIG["run_id"], "FINISHED")
    json.dumps(points(recorder), allow_nan=False)


@pytest.mark.parametrize(
    "invalid",
    [10**400, -(10**400), float("inf"), float("nan"), -1, True, "123"],
    ids=["overflow", "negative-overflow", "inf", "nan", "negative", "bool", "text"],
)
def test_invalid_checkpoint_sizes_do_not_interrupt_callbacks_or_publish_zero(recorder, invalid):
    recorder.event("on_save", context(metrics={"checkpoint_size_bytes": invalid}))
    recorder.event("on_log", context())
    recorder.close("FINISHED")
    values = {point["key"]: point["value"] for point in points(recorder)}
    assert "checkpoint/size_bytes" not in values
    assert values["checkpoint/success"] == 1
    assert values["checkpoint/count"] == 1
    assert values["train/loss"] == 0.25
    json.dumps(points(recorder), allow_nan=False)


def test_overflowing_checkpoint_size_is_accounted_and_falls_back_to_approved_scan(
    recorder, tmp_path
):
    root = tmp_path.resolve()
    (root / "weights").write_bytes(b"abc")
    recorder.set_artifact_root(str(root))
    recorder.event(
        "on_save",
        context(metrics={"checkpoint_size_bytes": 10**400, "checkpoint_path": str(root)}),
    )
    recorder.close()
    sizes = [
        point["value"] for point in points(recorder) if point["key"] == "checkpoint/size_bytes"
    ]
    assert sizes == [3]
    assert recorder.client.log.call_args.kwargs["metrics"]["tracking/nonfinite_values"] == 1


@pytest.mark.parametrize("size", [0, 7, 7.5])
def test_valid_checkpoint_size_is_preserved_without_directory_scan(recorder, tmp_path, size):
    root = tmp_path.resolve()
    recorder.set_artifact_root(str(root))
    recorder._directory_size = Mock(side_effect=AssertionError("Backend size is available"))
    recorder.event(
        "on_save", context(metrics={"checkpoint_size_bytes": size, "checkpoint_path": str(root)})
    )
    recorder.close()
    sizes = [
        point["value"] for point in points(recorder) if point["key"] == "checkpoint/size_bytes"
    ]
    assert sizes == [size]
    recorder._directory_size.assert_not_called()


@pytest.mark.parametrize(
    ("first", "second", "valid"),
    [
        ("a+b", "a?b", "a_b"),
        ("x" * 260 + "a", "x" * 260 + "b", "x" * 244),
        ("測" * 260 + "a", "測" * 260 + "b", "測" * 244),
    ],
    ids=["sanitized", "truncated-ascii", "truncated-unicode"],
)
def test_distinct_backend_keys_survive_sanitization_and_truncation(first, second, valid):
    backend = {first: 1, second: 2, valid: 3, "loss": 0.25}
    values = runtime.numeric_metrics({"train": backend})
    assert len(values) == 4
    assert set(values.values()) == {1, 2, 3, 0.25}
    assert values["train/loss"] == 0.25
    assert values["train/" + valid] == 3
    assert all(len(key) <= 250 for key in values)
    assert values == runtime.numeric_metrics({"train": dict(reversed(list(backend.items())))})
    assert values == runtime.numeric_metrics(backend, prefix="train/")
    for raw, number in ((first, 1), (second, 2)):
        name = next(iter(runtime.numeric_metrics({raw: number}, prefix="train")))
        assert values[name] == number


def test_valid_key_matching_a_generated_name_cannot_be_overwritten():
    raw = "train/a+b"
    generated = next(iter(runtime.numeric_metrics({raw: 1})))
    backend = {raw: 1, generated: 2, "train/loss": 0.25}
    values = runtime.numeric_metrics(backend)
    assert len(values) == 3
    assert values[generated] == 2
    assert values["train/loss"] == 0.25
    assert set(values.values()) == {1, 2, 0.25}
    assert values == runtime.numeric_metrics(dict(reversed(list(backend.items()))))


@pytest.mark.parametrize("status", ["FINISHED", "FAILED", "KILLED"])
@pytest.mark.parametrize("error_status", [408, 429, 500, 502, 503, 504])
def test_close_retries_transient_finish_once_without_repeating_drain_or_summaries(
    recorder, status, error_status
):
    recorder.client.finish.side_effect = [runtime.TrackingError(error_status), None]
    recorder._writer.join = Mock(wraps=recorder._writer.join)
    recorder._summary = Mock(wraps=recorder._summary)
    recorder._aggregate_delivery = Mock(wraps=recorder._aggregate_delivery)
    recorder.close(status)
    recorder.close("KILLED" if status != "KILLED" else "FINISHED")
    assert recorder.client.finish.call_args_list == [call(CONFIG["run_id"], status)] * 2
    recorder._writer.join.assert_called_once_with(timeout=recorder.DRAIN_TIMEOUT)
    recorder._summary.assert_called_once_with()
    recorder._aggregate_delivery.assert_called_once()
    assert recorder.client.log.call_count == 2
    assert recorder._queue.unfinished_tasks == 0


def test_close_stops_after_one_transient_retry_even_if_it_also_fails(recorder):
    recorder.client.finish.side_effect = runtime.TrackingError(503)
    recorder.close("FAILED")
    recorder.close("KILLED")
    assert recorder.client.finish.call_args_list == [call(CONFIG["run_id"], "FAILED")] * 2
    assert recorder.client.log.call_count == 2


@pytest.mark.parametrize("error_status", [400, 401, 403, 404, 409, 422])
def test_close_does_not_retry_permanent_finalization_errors(recorder, error_status):
    recorder.client.finish.side_effect = runtime.TrackingError(error_status)
    recorder.close("FINISHED")
    recorder.close("FAILED")
    recorder.client.finish.assert_called_once_with(CONFIG["run_id"], "FINISHED")
    assert recorder.client.log.call_count == 2


def test_close_does_not_repeat_acknowledged_finalization(recorder):
    recorder.close("FINISHED")
    recorder.close("FAILED")
    recorder.close("KILLED")
    recorder.client.finish.assert_called_once_with(CONFIG["run_id"], "FINISHED")
    assert recorder.client.log.call_count == 2


def test_close_without_status_only_drains_and_reports(recorder):
    recorder.close()
    recorder.client.finish.assert_not_called()
    assert recorder.client.log.call_count == 2
