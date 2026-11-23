"""Tracking boundaries, lifecycle measurements, and serialization regression tests."""

import base64
import importlib.util
import inspect
import json
import os
import shlex
import socket
import sys
import threading
import zlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from kubernetes.client.exceptions import ApiException
from pydantic import ValidationError

from training_service import jobs, ray_entrypoint
from training_service.contract import load_contract
from training_service.errors import ApiError
from training_service.metrics import client, runtime, tracking
from training_service_api.models.create_job_request import CreateJobRequest
from training_service_api.models.training_job import TrainingJob

RUN = "a" * 32
CONFIG = {
    "tracking_uri": "https://tracking.example/mlflow",
    "project": "team-alpha",
    "run_id": RUN,
    "experiment_id": "42",
    "experiment_name": "Default",
    "ca_bundle": "",
}


def sample():
    examples = load_contract()["paths"]["/projects/{project}/training-jobs"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]
    return deepcopy(examples["pvc_inputs_output"]["value"])


@pytest.mark.parametrize("name", ["", " ", "\t\n", "a" * 257, 12])
def test_invalid_experiment_names_rejected(name):
    payload = sample()
    payload["mlflow_experiment_name"] = name
    with pytest.raises(ValidationError):
        CreateJobRequest.model_validate(payload)


def test_experiment_optional_and_workspace_not_client_selectable():
    payload = sample()
    assert CreateJobRequest.model_validate(payload).mlflow_experiment_name is None
    payload["mlflow_experiment_name"] = "Fine tuning / model A"
    assert (
        CreateJobRequest.model_validate(payload).mlflow_experiment_name
        == payload["mlflow_experiment_name"]
    )
    for key in ("mlflow_workspace", "mlflow_tracking_uri"):
        invalid = {**payload, key: "other-project"}
        with pytest.raises(ValidationError):
            CreateJobRequest.model_validate(invalid)


@pytest.mark.parametrize(
    "uri",
    [
        "http://tracking.example",
        "https://user:pass@host",
        "https://host?q=1",
        "https://host/#x",
        "https://host:0",
    ],
)
def test_tracking_requires_verified_https_without_inline_credentials(uri):
    with pytest.raises(client.TrackingError):
        client.TrackingClient(uri, "team-alpha", lambda: "credential")


def test_transport_isolates_project_and_refreshes_token_without_global_state():
    tokens = iter(("first", "rotated"))
    tracking_client = client.TrackingClient(
        CONFIG["tracking_uri"], "team-alpha", lambda: next(tokens)
    )
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return b"{}"

    def open_request(request, timeout):
        requests.append(request)
        # Base timeout plus a small payload-scaled allowance.
        assert 5 <= timeout <= 10
        return Response()

    tracking_client._opener.open = open_request
    tracking_client.request("GET", "runs/get", {"run_id": RUN})
    tracking_client.request("POST", "runs/update", {"run_id": RUN, "status": "FINISHED"})
    assert [r.get_header("Authorization") for r in requests] == ["Bearer first", "Bearer rotated"]
    assert all(r.get_header("X-mlflow-workspace") == "team-alpha" for r in requests)
    assert requests[0].full_url.startswith(CONFIG["tracking_uri"] + "/api/2.0/mlflow/")


def test_transport_sanitizes_http_errors():
    tracking_client = client.TrackingClient(
        CONFIG["tracking_uri"], "team-alpha", lambda: "credential"
    )
    tracking_client._opener.open = Mock(
        side_effect=HTTPError("https://tracking.example", 403, "secret-response", {}, None)
    )
    with pytest.raises(client.TrackingError) as raised:
        tracking_client.request("POST", "runs/create", {})
    assert raised.value.status == 403
    assert "secret-response" not in str(raised.value)
    assert client._NoRedirect().redirect_request(None, None) is None


def test_default_experiment_and_concurrent_creation(monkeypatch):
    obj = client.TrackingClient(CONFIG["tracking_uri"], "team-alpha", lambda: "credential")
    request = Mock(
        side_effect=[
            client.TrackingError(404),
            client.TrackingError(409),
            {"experiment": {"experiment_id": "42"}},
        ]
    )
    monkeypatch.setattr(obj, "request", request)
    assert obj.experiment("Default") == "42"
    assert request.call_args_list[1].args == ("POST", "experiments/create", {"name": "Default"})


def test_discovery_checks_server_not_just_component_flag(monkeypatch):
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_ENABLED", "true")
    monkeypatch.setattr(tracking.config, "load_incluster_config", lambda **kwargs: None)
    custom = Mock()
    monkeypatch.setattr(tracking.client, "CustomObjectsApi", lambda api: custom)
    custom.get_cluster_custom_object.return_value = {
        "metadata": {"generation": 2},
        "status": {
            "url": CONFIG["tracking_uri"],
            "conditions": [{"type": "Available", "status": "True", "observedGeneration": 2}],
        },
    }
    assert tracking.discover() == CONFIG["tracking_uri"]
    custom.get_cluster_custom_object.return_value["status"]["conditions"][0][
        "observedGeneration"
    ] = 1
    with pytest.raises(ApiError, match="MLflow tracking is unavailable"):
        tracking.discover()
    custom.get_cluster_custom_object.side_effect = ApiException(status=404)
    assert tracking.discover() is None
    custom.get_cluster_custom_object.side_effect = ApiException(status=403)
    with pytest.raises(ApiError) as raised:
        tracking.discover()
    assert raised.value.status_code == 503


def test_training_identity_is_checked_in_target_project(monkeypatch):
    monkeypatch.setattr(tracking.config, "load_incluster_config", lambda **kwargs: None)
    auth = Mock()
    monkeypatch.setattr(tracking.client, "AuthorizationV1Api", lambda api: auth)
    auth.create_subject_access_review.return_value = SimpleNamespace(
        status=SimpleNamespace(allowed=True, evaluation_error="")
    )
    tracking.verify_training_access("team-alpha")
    for call in auth.create_subject_access_review.call_args_list:
        spec = call.args[0].spec
        # Only the exact project service-account group is queried; broader
        # identities would give false-positive preflights.
        assert spec.user is None
        assert spec.groups == ["system:serviceaccounts:team-alpha"]
        assert spec.resource_attributes.namespace == "team-alpha"
    auth.create_subject_access_review.return_value.status.allowed = False
    with pytest.raises(ApiError) as raised:
        tracking.verify_training_access("team-alpha")
    assert raised.value.status_code == 403


class FakeTracking:
    def __init__(self, *args, **kwargs):
        self.logs = []
        self.finished = []
        self.batches = []

    def log(self, run_id, **kwargs):
        self.logs.append((run_id, kwargs))

    def log_batch(self, run_id, metrics, tags):
        self.batches.append(metrics)
        self.log(run_id, tags=tags)
        for point in metrics:
            self.log(run_id, metrics={point["key"]: point["value"]}, step=point["step"])

    def get_run(self, run_id):
        tags = {}
        for _, call in self.logs:
            tags.update(call.get("tags", {}))
        return {"data": {"tags": [{"key": k, "value": v} for k, v in tags.items()]}}

    def finish(self, run_id, status):
        self.finished.append((run_id, status))


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(runtime, "TrackingClient", FakeTracking)
    record = runtime.RuntimeRecorder(CONFIG)
    monkeypatch.setattr(record, "start_sampling", lambda: None)
    yield record
    record.close()


def context(**kwargs):
    return SimpleNamespace(
        **{
            "step": 3,
            "epoch": 0.5,
            "loss": 0.75,
            "learning_rate": 0.0001,
            "is_main_process": True,
            "metrics": {},
            "output_dir": "",
            **kwargs,
        }
    )


def test_callbacks_preserve_training_eval_and_checkpoint_measurements(recorder, tmp_path):
    (tmp_path / "weights").write_bytes(b"abc")
    recorder.set_artifact_root(str(tmp_path))
    recorder.event("on_train_begin", context(step=7))
    recorder.event("on_step_begin", context())
    recorder.event("on_step_end", context())
    recorder.event("on_log", context(metrics={"grad_norm": 2, "tokens_per_second": 50}))
    recorder.event("on_evaluate", context(metrics={"accuracy": 0.8, "eval_loss": 1.5}))
    recorder.event(
        "on_save",
        context(
            metrics={
                "checkpoint_path": str(tmp_path),
                "checkpoint_duration_seconds": 1.2,
                "checkpoint_type": "jit",
            }
        ),
    )
    recorder.event("on_train_end", context())
    recorder.close("FINISHED")
    values = {
        key: value
        for _, call in recorder.client.logs
        for key, value in call.get("metrics", {}).items()
    }
    assert values["training/initial_step"] == 7
    assert values["train/loss"] == 0.75
    assert values["train/grad_norm"] == 2
    assert values["train/tokens_per_second"] == 50
    assert values["eval/accuracy"] == 0.8
    assert values["eval/loss"] == 1.5
    assert values["checkpoint/size_bytes"] == 3
    assert values["checkpoint/duration_seconds"] == 1.2
    assert values["training/step_wall_seconds"] >= 0
    assert recorder.client.finished == [(RUN, "FINISHED")]


def test_checkpoint_duration_is_not_invented(recorder):
    recorder.event("on_save", context())
    recorder.close()
    assert all(
        "checkpoint/duration_seconds" not in call.get("metrics", {})
        for _, call in recorder.client.logs
    )
    assert any(
        call.get("tags", {}).get("checkpoint.duration_source") == "not_exposed"
        for _, call in recorder.client.logs
    )


@pytest.mark.parametrize("storage_failure", [False, True])
def test_checkpoint_scan_runs_off_training_thread_and_tolerates_storage_loss(
    recorder, monkeypatch, tmp_path, storage_failure
):
    training_thread = threading.current_thread()
    observed = []
    recorder.set_artifact_root(str(tmp_path))
    original = os.scandir

    def scan(path):
        observed.append(threading.current_thread())
        if storage_failure:
            raise OSError("checkpoint was rotated")
        return original(path)

    monkeypatch.setattr(os, "scandir", scan)
    recorder.event("on_save", context(metrics={"checkpoint_path": str(tmp_path)}))
    recorder.close("FINISHED")
    assert observed and all(thread is not training_thread for thread in observed)
    assert recorder.client.finished == [(RUN, "FINISHED")]


def test_only_main_rank_emits_common_training_metrics(recorder):
    recorder.event("on_log", context(is_main_process=False))
    recorder.close()
    assert not any("train/loss" in call.get("metrics", {}) for _, call in recorder.client.logs)


def test_non_numeric_metrics_and_credentials_never_become_metric_payloads():
    result = runtime.numeric_metrics(
        {
            "loss": 1,
            "tokens_per_second": 42,
            "api_key": 3,
            "hf_token": "private",
            "nested": {"accuracy": 0.8},
            "nan": float("nan"),
            "enabled": True,
        }
    )
    assert result == {"loss": 1.0, "tokens_per_second": 42.0, "nested/accuracy": 0.8}


def test_resources_measure_pod_cgroup_not_host_memory(monkeypatch, tmp_path):
    for name, value in {
        "memory.current": "1024",
        "memory.peak": "2048",
        "cpu.stat": "usage_usec 2000000\nthrottled_usec 500000\n",
        "io.stat": "1:0 rbytes=100 wbytes=200 rios=1\n",
    }.items():
        (tmp_path / name).write_text(value)
    sampler = runtime.SystemSampler()
    monkeypatch.setattr(sampler, "root", tmp_path)
    assert sampler.sample() == {
        "memory_bytes": 1024.0,
        "memory_peak_bytes": 2048.0,
        "cpu_usage_seconds": 2.0,
        "cpu_throttled_seconds": 0.5,
        "storage_read_bytes": 100.0,
        "storage_write_bytes": 200.0,
    }


def tracked_job(uri=CONFIG["tracking_uri"]):
    obj = client.TrackingClient(uri, "team-alpha", lambda: "never-read")
    run = tracking.TrackedRun(obj, RUN, "42", "Default")
    return {
        "metadata": {
            "name": "job",
            "uid": "rayjob-uid",
            "creationTimestamp": "2026-11-19T00:00:00Z",
            "annotations": {
                tracking.ANNOTATION: json.dumps(
                    {**run.metadata(), "tracking_uri": uri, "project": "team-alpha"}
                )
            },
        }
    }


def bound_response(status="RUNNING", **tags):
    return {
        "info": {"run_id": RUN, "experiment_id": "42", "status": status},
        "data": {
            "params": [{"key": "training.rayjob_uid", "value": "rayjob-uid"}],
            "tags": [
                {"key": k, "value": v}
                for k, v in {
                    "training.project": "team-alpha",
                    "training.job_id": "job",
                    **tags,
                }.items()
            ],
        },
    }


def test_get_job_returns_only_authenticated_project_scoped_link(monkeypatch):
    obj = Mock()
    obj.get_run.return_value = bound_response()
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    job = tracked_job()
    assert "mlflow_url" not in jobs.detail(job, "team-alpha")
    result = jobs.detail(job, "team-alpha") | tracking.public_metadata(
        job, "team-alpha", "caller", None
    )
    assert TrainingJob.model_validate(result).mlflow_run_id == RUN
    assert tracking.metadata(job, "different-project") is None
    job["metadata"]["uid"] = "copied-job-uid"
    assert tracking.public_metadata(job, "team-alpha", "caller", None) == {}


@pytest.mark.parametrize("status", ["FINISHED", "FAILED", "KILLED"])
def test_deletion_preserves_terminal_mlflow_history(monkeypatch, status):
    obj = Mock()
    obj.get_run.return_value = bound_response(status)
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    tracking.finalize_deleted(tracked_job(), "team-alpha", "caller-credential", None)
    obj.log.assert_not_called()
    obj.finish.assert_not_called()


@pytest.mark.parametrize(
    "state, expected", [("PENDING", "KILLED"), ("SUCCEEDED", "FINISHED"), ("FAILED", "FAILED")]
)
def test_deletion_finalizes_only_unfinished_runs(monkeypatch, state, expected):
    obj = Mock()
    obj.get_run.return_value = bound_response()
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    monkeypatch.setattr(tracking, "TrackingClient", lambda *args, **kwargs: obj)
    job = tracked_job()
    job["status"] = {"jobStatus": state}
    tracking.finalize_deleted(job, "team-alpha", "caller-credential", None)
    obj.finish.assert_called_once_with(RUN, expected)


def test_deletion_never_sends_caller_token_to_annotation_selected_server(monkeypatch):
    job = tracked_job("https://untrusted.example/mlflow")
    monkeypatch.setattr(tracking, "discover", lambda **kwargs: CONFIG["tracking_uri"])
    constructor = Mock()
    monkeypatch.setattr(tracking, "TrackingClient", constructor)
    tracking.finalize_deleted(job, "team-alpha", "caller-credential", None)
    assert tracking.public_metadata(job, "team-alpha", "caller", None) == {}
    constructor.assert_not_called()


def test_serialized_callback_has_no_module_global_dependencies(monkeypatch):
    base = type("TrainingHubCallback", (), {})
    monkeypatch.setitem(
        sys.modules, "training_hub.callbacks", SimpleNamespace(TrainingHubCallback=base)
    )
    path = Path(runtime.__file__).with_name("callback.py")
    spec = importlib.util.spec_from_file_location("_metrics_callback_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    source = inspect.getsource(module.MLflowMetricsCallback)
    namespace = {"TrainingHubCallback": base}
    exec(source, namespace)
    record = Mock()
    monkeypatch.setitem(
        sys.modules,
        "training_service_metrics_runtime",
        SimpleNamespace(current_recorder=lambda: record),
    )
    namespace["MLflowMetricsCallback"]().on_log(context())
    assert record.event.call_args.args[0] == "on_log"
    monkeypatch.setenv(
        "TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS", f"{socket.gethostname()}:{os.getpid()}"
    )
    monkeypatch.setenv("RANK", "0")
    callback = namespace["MLflowMetricsCallback"]()
    callback.on_train_end(context())
    record.close.assert_not_called()
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS", "different-pod:1")
    callback.on_train_end(context())
    record.close.assert_called_once()


@pytest.mark.parametrize("algorithm", ["sft", "osft", "lora_sft", "lora_grpo"])
@pytest.mark.parametrize("failed", [False, True])
def test_every_algorithm_gets_common_run_lifetime(monkeypatch, algorithm, failed):
    record = Mock()
    monkeypatch.setitem(
        sys.modules,
        "training_service_metrics_runtime",
        SimpleNamespace(current_recorder=lambda: record),
    )
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_RUN", "{}")
    execute = Mock(side_effect=ValueError("training failed") if failed else None)
    monkeypatch.setattr(ray_entrypoint, "_execute", execute)
    if failed:
        with pytest.raises(ValueError):
            ray_entrypoint.main({"algorithm": algorithm, "mlflow": CONFIG})
    else:
        ray_entrypoint.main({"algorithm": algorithm, "mlflow": CONFIG})
    assert record.close.call_args.args[0] == ("FAILED" if failed else "FINISHED")


@pytest.mark.parametrize("supports_callbacks", [False, True])
def test_actual_training_call_attaches_callbacks_only_when_runtime_supports_them(
    monkeypatch, tmp_path, supports_callbacks
):
    def with_callbacks(callbacks=None):
        pass

    def without_callbacks():
        pass

    algorithm = SimpleNamespace(train=Mock(return_value={"loss": 1.0}))
    record = Mock()
    callback = object()
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_RUN", json.dumps(CONFIG))
    monkeypatch.setitem(
        sys.modules,
        "training_hub",
        SimpleNamespace(
            create_algorithm=lambda *args: algorithm,
            sft=with_callbacks if supports_callbacks else without_callbacks,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "training_service_metrics_runtime",
        SimpleNamespace(current_recorder=lambda: record),
    )
    monkeypatch.setitem(
        sys.modules,
        "training_service_metrics_callback",
        SimpleNamespace(MLflowMetricsCallback=lambda: callback),
    )
    backend = ray_entrypoint.TrainingBackend()
    backend.name = "sft"
    assert backend.run("model", "data", tmp_path, {}, 1, 0) == {"loss": 1.0}
    options = algorithm.train.call_args.kwargs
    assert options.get("callbacks") == ([callback] if supports_callbacks else None)
    assert (
        record.emit.call_args.kwargs["tags"]["tracking.unified_callbacks"]
        == str(supports_callbacks).lower()
    )


def test_embedded_metrics_program_compiles():
    command = shlex.split(jobs._entrypoint({"mlflow": CONFIG}))
    body = zlib.decompress(base64.b64decode(command[2].split('"')[1])).decode()
    compile(body, "<ray-metrics-entrypoint>", "exec")
    assert "training_service_metrics_callback" in body


def test_portable_metrics_do_not_exceed_linux_exec_argument_limit():
    configuration = {"mlflow": CONFIG, "parameters": {"metadata": os.urandom(30000).hex()}}
    command = shlex.split(jobs._entrypoint(configuration))
    assert len(command[2].encode()) < 128 * 1024


def test_oversized_entrypoint_is_rejected_before_ray_submission():
    with pytest.raises(ApiError, match="generated training command exceeds"):
        jobs._entrypoint({"mlflow": CONFIG, "parameters": {"metadata": os.urandom(150000).hex()}})


def test_runtime_reconstructs_full_trust_bundle_from_environment(monkeypatch):
    from requests.certs import where

    bundle = Path(where()).read_text()
    parts = [bundle[offset : offset + 16384] for offset in range(0, len(bundle), 16384)]
    for index, value in enumerate(parts):
        monkeypatch.setenv(f"TRAINING_SERVICE_MLFLOW_CA_{index}", value)
    constructor = Mock(return_value=FakeTracking())
    monkeypatch.setattr(runtime, "TrackingClient", constructor)
    record = runtime.RuntimeRecorder({**CONFIG, "ca_bundle_parts": str(len(parts))})
    record.close()
    assert constructor.call_args.kwargs["ca_bundle"] == bundle


def test_missing_trust_part_does_not_silently_drop_tls_configuration(monkeypatch):
    monkeypatch.delenv("TRAINING_SERVICE_MLFLOW_CA_0", raising=False)
    constructor = Mock()
    monkeypatch.setattr(runtime, "TrackingClient", constructor)
    with pytest.raises(KeyError):
        runtime.RuntimeRecorder({**CONFIG, "ca_bundle_parts": "1"})
    constructor.assert_not_called()
