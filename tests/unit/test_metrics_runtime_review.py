"""Focused runtime regressions from the MLflow integration review."""

import json
import os
import signal
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training_service import ray_entrypoint
from training_service.metrics import runtime

CONFIG = {"tracking_uri": "https://tracking.example", "project": "team", "run_id": "a" * 32}
GPU_A = "GPU-11111111-1111-1111-1111-111111111111"
GPU_B = "GPU-22222222-2222-2222-2222-222222222222"


class RecordingClient:
    def __init__(self):
        self.tags = {}
        self.points = []
        self.logs = []
        self.finished = []
        self._lock = threading.Lock()

    def log_batch(self, run_id, metrics, tags):
        with self._lock:
            self.points.extend(metrics)
            self.tags.update(tags)

    def log(self, run_id, **kwargs):
        with self._lock:
            self.tags.update(kwargs.get("tags", {}))
            self.logs.append(kwargs)

    def get_run(self, run_id):
        with self._lock:
            return {"data": {"tags": [{"key": k, "value": v} for k, v in self.tags.items()]}}

    def finish(self, run_id, status):
        self.finished.append((run_id, status))


@pytest.fixture
def make_recorder(monkeypatch):
    records = []
    monkeypatch.delenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    def make(configuration=None, client=None):
        obj = client or RecordingClient()
        monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: obj)
        record = runtime.RuntimeRecorder({**CONFIG, **(configuration or {})})
        records.append(record)
        return record

    yield make
    for record in records:
        record.close()
        record._writer.join(2)
        assert not record._writer.is_alive()


def summary(**kwargs):
    return {
        "state": "complete",
        "failed": 0,
        "unresolved": 0,
        "nonfinite": 0,
        "rank": None,
        "world_size": 1,
        **kwargs,
    }


def block_writer(client):
    ready, release = threading.Event(), threading.Event()
    original = client.log_batch

    def blocked(*args):
        ready.set()
        assert release.wait(3)
        original(*args)

    client.log_batch = blocked
    return ready, release


def scan_values(record):
    return [point["value"] for point in record.client.points if point["key"] == "scan/bytes"]


def test_idle_writer_exits_without_using_drain_deadline(make_recorder):
    record = make_recorder()
    record._queue.join()
    record._stop.set()
    record._writer.join(0.5)
    assert not record._writer.is_alive()


def test_sigterm_before_sampler_start_still_finalizes_real_recorder(make_recorder, monkeypatch):
    record = make_recorder()
    monkeypatch.setitem(sys.modules, "training_service_metrics_runtime", runtime)
    monkeypatch.setattr(runtime, "_CURRENT", record)
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_RUN", json.dumps(record.configuration))
    monkeypatch.setenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS", "review-driver")
    execute = Mock()
    monkeypatch.setattr(ray_entrypoint, "_execute", execute)
    original_start = threading.Thread.start

    def start(thread):
        if thread.name == "mlflow-resources":
            signal.raise_signal(signal.SIGTERM)
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as raised:
        ray_entrypoint.main({"mlflow": CONFIG})

    assert raised.value.code == 128 + signal.SIGTERM
    assert record._sampler is not None and record._sampler.ident is None
    assert not record._writer.is_alive()
    assert record.client.finished == [(CONFIG["run_id"], "KILLED")]
    assert record.client.tags["training.phase"] == "cancelled"
    assert signal.getsignal(signal.SIGTERM) == previous
    execute.assert_not_called()


def test_failed_batch_accounting_has_no_double_counted_shutdown_snapshot(
    make_recorder, monkeypatch
):
    record = make_recorder()
    record._queue.join()
    snapshots = []

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, *args):
            if threading.current_thread() is record._writer:
                snapshots.append(record._failed + record._accepted - record._completed)
            self.lock.release()

    record._lock = ObservedLock()
    record.client.log_batch = Mock(side_effect=runtime.TrackingError())
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    record.emit({"train/loss": 0.5})
    record._queue.join()
    assert snapshots and snapshots == [1] * len(snapshots)
    assert record._summary() == summary(state="incomplete", failed=1)
    record.close("FINISHED")
    assert record.client.logs[-1]["metrics"]["tracking/undelivered_events"] == 1


@pytest.mark.parametrize("rank", [[], {}, 1, True])
@pytest.mark.parametrize("suffix", ["", ".terminal"])
def test_malformed_producer_rank_cannot_prevent_terminal_status(make_recorder, rank, suffix):
    record = make_recorder()
    record._queue.join()
    record.client.tags.update(
        {
            "tracking.unified_callbacks": "true",
            "tracking.producer.peer" + suffix: json.dumps(summary(rank=rank)),
        }
    )
    record.close("FINISHED")
    assert record.client.finished == [(CONFIG["run_id"], "FINISHED")]
    assert "tracking.delivery" not in record.client.tags


@pytest.mark.parametrize("rank", [None, "0"])
def test_null_and_string_producer_ranks_are_accepted(make_recorder, rank):
    record = make_recorder()
    record._queue.join()
    record.client.tags["tracking.producer.peer"] = json.dumps(summary(rank=rank))
    record.close("FINISHED")
    assert record.client.tags["tracking.delivery"] == "complete"


@pytest.mark.parametrize("terminal_first", [False, True])
def test_terminal_preferred_over_startup_without_double_counting(make_recorder, terminal_first):
    record = make_recorder()
    record._queue.join()
    startup = ("tracking.producer.peer.startup", json.dumps(summary(state="running", failed=10)))
    terminal = (
        "tracking.producer.peer.terminal",
        json.dumps(summary(state="incomplete", failed=2, unresolved=3, nonfinite=4)),
    )
    record.client.tags.update([terminal, startup] if terminal_first else [startup, terminal])
    record.close()
    result = record.client.logs[-1]
    assert result["tags"]["tracking.delivery"] == "incomplete"
    assert result["metrics"] == {
        "tracking/failed_events": 2,
        "tracking/unresolved_events": 3,
        "tracking/undelivered_events": 5,
        "tracking/nonfinite_values": 4,
    }


def test_complete_terminal_wins_over_late_running_startup(make_recorder):
    record = make_recorder()
    record._queue.join()
    record.client.tags.update(
        {
            "tracking.producer.peer.terminal": json.dumps(summary(rank="0")),
            "tracking.producer.peer.startup": json.dumps(summary(state="running")),
            "tracking.unified_callbacks": "true",
        }
    )
    record.close()
    assert record.client.tags["tracking.delivery"] == "complete"


@pytest.mark.parametrize("state", ["running", "complete"])
def test_startup_without_terminal_remains_incomplete(make_recorder, state):
    record = make_recorder()
    record._queue.join()
    record.client.tags["tracking.producer.peer.startup"] = json.dumps(summary(state=state))
    record.close()
    assert record.client.tags["tracking.delivery"] == "incomplete"


def test_failed_terminal_summary_write_cannot_report_complete_delivery(make_recorder):
    record = make_recorder()
    record._queue.join()
    original = record.client.log

    def log(run_id, **kwargs):
        if record._producer + ".terminal" in kwargs.get("tags", {}):
            raise runtime.TrackingError()
        original(run_id, **kwargs)

    record.client.log = log
    record.close("FINISHED")
    assert record.client.tags["tracking.delivery"] == "incomplete"
    assert record.client.finished == [(CONFIG["run_id"], "FINISHED")]


def test_valid_terminal_supersedes_malformed_obsolete_startup(make_recorder):
    record = make_recorder()
    record._queue.join()
    record.client.tags.update(
        {
            "tracking.producer.peer.startup": json.dumps(summary(rank=[])),
            "tracking.producer.peer.terminal": json.dumps(summary()),
        }
    )
    record.close()
    assert record.client.tags["tracking.delivery"] == "complete"


@pytest.mark.parametrize("state", ["complete", "running", "incomplete"])
def test_legacy_single_producer_tags_still_contribute(make_recorder, state):
    record = make_recorder()
    record._queue.join()
    record.client.tags["tracking.producer.legacy"] = json.dumps(summary(state=state))
    record.close()
    expected = "complete" if state == "complete" else "incomplete"
    assert record.client.tags["tracking.delivery"] == expected


def test_delayed_startup_cannot_overwrite_shutdown_summary(make_recorder, monkeypatch):
    obj = RecordingClient()
    ready, release = block_writer(obj)
    record = make_recorder(client=obj)
    assert ready.wait(2)
    monkeypatch.setattr(record, "DRAIN_TIMEOUT", 0.01)
    try:
        record.close()
        terminal_key = record._producer + ".terminal"
        assert json.loads(obj.tags[terminal_key])["unresolved"] == 1
    finally:
        release.set()
        record._writer.join(2)
    assert json.loads(obj.tags[record._producer + ".startup"])["state"] == "running"
    assert json.loads(obj.tags[terminal_key])["unresolved"] == 1
    driver = make_recorder(client=obj)
    driver.close()
    assert obj.logs[-1]["metrics"]["tracking/unresolved_events"] == 1
    assert obj.tags["tracking.delivery"] == "incomplete"


def test_unapproved_scan_never_inspects_callback_path(make_recorder, monkeypatch, tmp_path):
    record = make_recorder()
    opened = Mock(side_effect=AssertionError("An unapproved path must not be inspected"))
    monkeypatch.setattr(record, "_open_artifact_directory", opened)
    record.emit(directory_size=("scan/bytes", str(tmp_path)))
    record.close()
    opened.assert_not_called()
    assert scan_values(record) == []


def test_approved_scan_ignores_links_and_runs_off_training_thread(
    make_recorder, monkeypatch, tmp_path
):
    tree = tmp_path.resolve()
    root, outside = tree / "artifacts", tree / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside data")
    (root / "weights").write_bytes(b"abc")
    (root / "nested").mkdir()
    (root / "nested" / "weights").write_bytes(b"de")
    (root / "file_link").symlink_to(outside / "secret")
    (root / "directory_link").symlink_to(outside, target_is_directory=True)
    record = make_recorder()
    record.set_artifact_root(str(root))
    assert record.configuration["artifact_root"] == str(root)
    original_scandir, original_open = os.scandir, os.open
    scanned = []

    def scandir(descriptor):
        assert isinstance(descriptor, int)
        assert os.fstat(descriptor).st_ino != outside.stat().st_ino
        scanned.append(threading.current_thread())
        return original_scandir(descriptor)

    def open_directory(path, flags, *args, **kwargs):
        assert flags & os.O_NOFOLLOW
        assert flags & os.O_DIRECTORY
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(runtime.os, "scandir", scandir)
    monkeypatch.setattr(runtime.os, "open", open_directory)
    record.emit(directory_size=("scan/bytes", str(root)))
    record.close()
    assert scan_values(record) == [5]
    assert scanned and all(thread is record._writer for thread in scanned)


def test_approved_root_configuration_survives_subprocess_serialization(make_recorder, tmp_path):
    root = tmp_path.resolve()
    (root / "weights").write_bytes(b"abc")
    parent = make_recorder()
    parent.set_artifact_root(str(root))
    child = make_recorder(json.loads(json.dumps(parent.configuration)))
    child.emit(directory_size=("scan/bytes", str(root)))
    child.close()
    assert scan_values(child) == [3]


def test_approving_artifact_root_does_not_mutate_caller_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "TrackingClient", lambda *args, **kwargs: RecordingClient())
    configuration = dict(CONFIG)
    record = runtime.RuntimeRecorder(configuration)
    try:
        record.set_artifact_root(str(tmp_path.resolve()))
        assert configuration == CONFIG
        assert record.configuration["artifact_root"] == str(tmp_path.resolve())
    finally:
        record.close()


@pytest.mark.parametrize("target", ["outside", "traversal", "link", "ancestor_link"])
def test_callback_scan_cannot_escape_approved_root(make_recorder, tmp_path, target):
    tree = tmp_path.resolve()
    root, outside = tree / "artifacts", tree / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside data")
    (root / "link").symlink_to(outside, target_is_directory=True)
    (root / "ancestor").symlink_to(tree, target_is_directory=True)
    directory = {
        "outside": str(outside),
        "traversal": str(root / ".." / "outside"),
        "link": str(root / "link"),
        "ancestor_link": str(root / "ancestor" / "outside"),
    }[target]
    record = make_recorder({"artifact_root": str(root)})
    record.emit(directory_size=("scan/bytes", directory))
    record.close()
    assert scan_values(record) == []


def test_symlink_cannot_be_approved_as_artifact_root(make_recorder, tmp_path):
    tree = tmp_path.resolve()
    (tree / "link").symlink_to(tree, target_is_directory=True)
    record = make_recorder({"artifact_root": str(tree / "link")})
    record.emit(directory_size=("scan/bytes", str(tree / "link")))
    record.close()
    assert scan_values(record) == []


@pytest.mark.parametrize(
    "replacement", ["removed", "directory", "symlink", "root_symlink", "root_directory"]
)
def test_rotated_or_swapped_checkpoint_is_omitted(make_recorder, tmp_path, replacement):
    tree = tmp_path.resolve()
    root, outside = tree / "artifacts", tree / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside data")
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(b"abc")
    obj = RecordingClient()
    ready, release = block_writer(obj)
    record = make_recorder({"artifact_root": str(root)}, client=obj)
    assert ready.wait(2)
    try:
        record.emit(directory_size=("scan/bytes", str(checkpoint)))
        if replacement.startswith("root_"):
            root.rename(tree / "old_artifacts")
            if replacement == "root_symlink":
                root.symlink_to(outside, target_is_directory=True)
            else:
                checkpoint.mkdir(parents=True)
                (checkpoint / "other_weights").write_bytes(b"replacement")
        else:
            checkpoint.rename(root / "old_checkpoint")
            if replacement == "directory":
                checkpoint.mkdir()
                (checkpoint / "other_weights").write_bytes(b"replacement")
            elif replacement == "symlink":
                checkpoint.symlink_to(outside, target_is_directory=True)
    finally:
        release.set()
    record.close()
    assert scan_values(record) == []


def test_directory_swapped_to_symlink_during_scan_is_never_followed(
    make_recorder, monkeypatch, tmp_path
):
    tree = tmp_path.resolve()
    root, outside = tree / "artifacts", tree / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside data")
    nested = root / "nested"
    nested.mkdir()
    (nested / "weights").write_bytes(b"abc")
    record = make_recorder({"artifact_root": str(root)})
    original_open, original_scandir = os.open, os.scandir
    swapped = []
    outside_inode = outside.stat().st_ino

    def open_directory(path, flags, *args, **kwargs):
        if path == "nested" and threading.current_thread() is record._writer and not swapped:
            nested.rename(root / "old_nested")
            nested.symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return original_open(path, flags, *args, **kwargs)

    def scandir(descriptor):
        assert os.fstat(descriptor).st_ino != outside_inode
        return original_scandir(descriptor)

    monkeypatch.setattr(runtime.os, "open", open_directory)
    monkeypatch.setattr(runtime.os, "scandir", scandir)
    record.emit(directory_size=("scan/bytes", str(root)))
    record.close()
    assert swapped
    assert scan_values(record) == []


@pytest.mark.parametrize(
    ("limit", "value"), [("SCAN_MAX_ENTRIES", 1), ("SCAN_MAX_DEPTH", 0), ("SCAN_TIMEOUT", 0)]
)
def test_bounded_scans_omit_partial_sizes(make_recorder, monkeypatch, tmp_path, limit, value):
    root = tmp_path.resolve()
    (root / "weights").write_bytes(b"abc")
    (root / "nested").mkdir()
    (root / "nested" / "weights").write_bytes(b"def")
    record = make_recorder({"artifact_root": str(root)})
    monkeypatch.setattr(record, limit, value)
    record.emit(directory_size=("scan/bytes", str(root)))
    record.close()
    assert scan_values(record) == []


@pytest.fixture
def gpu_query(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    command = Mock(return_value=SimpleNamespace(returncode=0, stdout="10,20,40,50\n"))
    monkeypatch.setattr(runtime.subprocess, "run", command)
    return command


@pytest.mark.parametrize("name", ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"])
@pytest.mark.parametrize("value", ["", "none", "None", "void", "all", "-1", "0,all", "MIG-test"])
def test_disabled_or_unbounded_gpu_visibility_never_queries_node(
    gpu_query, monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    other = "CUDA_VISIBLE_DEVICES" if name == "NVIDIA_VISIBLE_DEVICES" else "NVIDIA_VISIBLE_DEVICES"
    monkeypatch.setenv(other, GPU_A)
    assert runtime.SystemSampler().gpu() == {}
    gpu_query.assert_not_called()


def test_gpu_less_head_without_visibility_never_runs_nvidia_smi(gpu_query):
    assert runtime.SystemSampler().gpu() == {}
    gpu_query.assert_not_called()


@pytest.mark.parametrize(
    ("cuda", "nvidia", "selected"),
    [
        (None, "3", "3"),
        (None, GPU_A, GPU_A),
        (GPU_A, None, GPU_A),
        ("0", GPU_A, GPU_A),
        ("0", "3", "3"),
        ("0,1", "3,5", "3,5"),
        ("0,1", f"{GPU_A},{GPU_B}", f"{GPU_A},{GPU_B}"),
        (GPU_B, f"{GPU_A},{GPU_B}", GPU_B),
    ],
)
def test_gpu_query_uses_explicit_allocation_selectors(
    gpu_query, monkeypatch, cuda, nvidia, selected
):
    if cuda is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
    if nvidia is not None:
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", nvidia)
    gpu_query.return_value.stdout = "10,20,40,50\n" * len(selected.split(","))
    values = runtime.SystemSampler().gpu()
    assert "--id=" + selected in gpu_query.call_args.args[0]
    assert values["gpu_0/utilization_percent"] == 10
    assert values["gpu_0/memory_used_bytes"] == 20 * 2**20
    assert values["gpu_0/memory_total_bytes"] == 40 * 2**20
    assert values["gpu_0/power_watts"] == 50
    assert len(values) == 4 * len(selected.split(","))


@pytest.mark.parametrize(
    ("cuda", "nvidia"),
    [
        ("0", None),
        ("1", "3"),
        ("0", f"{GPU_A},{GPU_B}"),
        ("1", "3,5"),
        (GPU_A, GPU_B),
        (GPU_A, "3"),
        ("0," + GPU_A, GPU_A),
    ],
)
def test_ambiguous_cuda_index_or_uuid_mapping_is_omitted(gpu_query, monkeypatch, cuda, nvidia):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
    if nvidia is not None:
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", nvidia)
    assert runtime.SystemSampler().gpu() == {}
    gpu_query.assert_not_called()


def test_gpu_query_rejects_unexpected_extra_devices(gpu_query, monkeypatch):
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", GPU_A)
    gpu_query.return_value.stdout *= 2
    assert runtime.SystemSampler().gpu() == {}
