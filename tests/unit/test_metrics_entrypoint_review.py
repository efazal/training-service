"""Regressions for recorder startup cancellation and artifact-root propagation."""

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training_service import ray_entrypoint

TRACKED = {"run_id": "synthetic", "tracking_uri": "https://tracking.example", "project": "test"}


@pytest.fixture
def entrypoint(monkeypatch):
    record = SimpleNamespace(emit=Mock(), start_sampling=Mock(), close=Mock())
    runtime = SimpleNamespace(current_recorder=Mock(return_value=record), _CURRENT=None)
    monkeypatch.setitem(sys.modules, "training_service_metrics_runtime", runtime)
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_RUN", "{}")
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS", "test-process")
    execute = Mock()
    monkeypatch.setattr(ray_entrypoint, "_execute", execute)
    return record, runtime, execute


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("stage", ["constructing", "published", "initial_event", "sampling"])
def test_startup_failure_closes_available_recorder_and_restores_signal(
    entrypoint, stage, failure_type
):
    record, runtime, execute = entrypoint
    failure = failure_type("startup interrupted")

    def construct():
        if stage == "published":
            runtime._CURRENT = record
        raise failure

    if stage in {"constructing", "published"}:
        runtime.current_recorder.side_effect = construct
    elif stage == "initial_event":
        record.emit.side_effect = [failure, None]
    else:
        record.start_sampling.side_effect = failure

    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(failure_type) as raised:
        ray_entrypoint.main({"mlflow": TRACKED})

    assert raised.value is failure
    assert signal.getsignal(signal.SIGTERM) == previous
    runtime.current_recorder.assert_called_once_with()
    execute.assert_not_called()
    if stage == "constructing":
        record.emit.assert_not_called()
        record.close.assert_not_called()
    else:
        killed = failure_type is KeyboardInterrupt
        record.close.assert_called_once_with("KILLED" if killed else "FAILED")
        assert record.emit.call_args.kwargs["tags"]["training.phase"] == (
            "cancelled" if killed else "failed"
        )


def test_startup_telemetry_failure_cannot_skip_close(entrypoint):
    record, runtime, execute = entrypoint
    failure = RuntimeError("telemetry failed")
    record.emit.side_effect = failure
    previous = signal.getsignal(signal.SIGTERM)

    with pytest.raises(RuntimeError, match="telemetry failed"):
        ray_entrypoint.main({"mlflow": TRACKED})

    record.close.assert_called_once_with("FAILED")
    runtime.current_recorder.assert_called_once_with()
    execute.assert_not_called()
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("stage", ["constructing", "published", "initial_event", "sampling"])
def test_real_sigterm_during_recorder_startup(stage):
    # Use the real current_recorder, including publication before atexit.register,
    # with a recorder that never starts threads, invokes hardware, or uses the network.
    script = """
import atexit
import os
import signal
import sys
from training_service import ray_entrypoint
from training_service.metrics import runtime

stage = sys.argv[1]
def terminate():
    os.kill(os.getpid(), signal.SIGTERM)

class Recorder:
    def __init__(self, configuration):
        print('CONSTRUCT', flush=True)
        if stage == 'constructing':
            terminate()

    def emit(self, metrics=None, *, tags=None):
        if stage == 'initial_event' and metrics and 'job/started_at_unix_seconds' in metrics:
            terminate()
        print('PHASE:' + tags['training.phase'], flush=True)

    def start_sampling(self):
        if stage == 'sampling':
            terminate()

    def close(self, status):
        print('CLOSED:' + status, flush=True)

def register(callback):
    if stage == 'published':
        assert runtime._CURRENT is not None
        terminate()
    return callback

runtime.RuntimeRecorder = Recorder
runtime._CURRENT = None
atexit.register = register
sys.modules['training_service_metrics_runtime'] = runtime
ray_entrypoint._execute = lambda cfg: print('UNEXPECTED_TRAINING', flush=True)
previous = signal.getsignal(signal.SIGTERM)
try:
    ray_entrypoint.main({'mlflow': {'run_id': 'synthetic'}})
finally:
    print('RESTORED:' + str(signal.getsignal(signal.SIGTERM) == previous), flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, stage],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 128 + signal.SIGTERM, result.stderr
    assert result.stdout.count("CONSTRUCT\n") == 1
    assert "RESTORED:True" in result.stdout
    assert "UNEXPECTED_TRAINING" not in result.stdout
    if stage == "constructing":
        assert "CLOSED:" not in result.stdout
    else:
        assert result.stdout.count("CLOSED:KILLED") == 1
        assert "PHASE:cancelled" in result.stdout


@pytest.fixture
def execution(monkeypatch, tmp_path):
    record = SimpleNamespace(configuration={**TRACKED, "expected_ranks": "2"}, emit=Mock())

    def set_artifact_root(root):
        assert Path(root).is_dir()
        record.configuration["artifact_root"] = root

    record.set_artifact_root = Mock(side_effect=set_artifact_root)
    monkeypatch.setitem(
        sys.modules,
        "training_service_metrics_runtime",
        SimpleNamespace(current_recorder=lambda: record),
    )
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_RUN", json.dumps(TRACKED))
    monkeypatch.setattr(
        ray_entrypoint,
        "Path",
        lambda path: tmp_path / "work" if path == "/tmp/training-api" else Path(path),
    )
    monkeypatch.setattr(ray_entrypoint, "_input", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(ray_entrypoint, "_ensure_s3_runtime", Mock())
    upload = Mock()
    monkeypatch.setattr(ray_entrypoint, "_upload_s3", upload)
    configuration = {
        "algorithm": "lora_sft",
        "model": {"scheme": "pvc"},
        "dataset": {"scheme": "pvc"},
        "parameters": {},
        "gpus_per_node": 1,
        "num_workers": 0,
        "mlflow": TRACKED,
    }
    return configuration, record, upload


@pytest.mark.parametrize("output_scheme", ["pvc", "s3", None])
def test_artifact_root_reaches_callback_subprocess_before_training(
    monkeypatch, tmp_path, execution, output_scheme
):
    configuration, record, upload = execution
    output_dir = tmp_path / "output" if output_scheme == "pvc" else tmp_path / "work/output"
    if output_scheme is not None:
        configuration["output"] = {"scheme": output_scheme, "mount_path": str(output_dir)}

    def train(*args):
        record.set_artifact_root.assert_called_once_with(str(output_dir))
        assert args[3] == output_dir
        child = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import os; print(os.environ['TRAINING_SERVICE_MLFLOW_RUN'])",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        assert json.loads(child.stdout) == {
            **record.configuration,
            "artifact_root": str(output_dir),
        }
        return {"loss": 0.5}

    training = Mock(side_effect=train)
    monkeypatch.setattr(ray_entrypoint, "_run_training", training)
    ray_entrypoint._execute(configuration)

    training.assert_called_once()
    record.emit.assert_any_call({"result": {"loss": 0.5}})
    record.emit.assert_any_call(
        directory_size=("artifacts/output_directory_bytes", str(output_dir))
    )
    if output_scheme == "s3":
        upload.assert_called_once_with(configuration["output"], output_dir)
    else:
        upload.assert_not_called()


def test_final_results_are_recorded_before_failed_s3_upload(monkeypatch, execution):
    configuration, record, upload = execution
    configuration["output"] = {"scheme": "s3"}
    monkeypatch.setattr(ray_entrypoint, "_run_training", Mock(return_value={"loss": 0.5}))

    def fail_upload(*args):
        record.emit.assert_any_call({"result": {"loss": 0.5}})
        size_metric, _ = record.emit.call_args.kwargs["directory_size"]
        assert size_metric == "artifacts/output_directory_bytes"
        raise OSError("upload failed")

    upload.side_effect = fail_upload
    with pytest.raises(OSError, match="upload failed"):
        ray_entrypoint._execute(configuration)
    upload.assert_called_once()
