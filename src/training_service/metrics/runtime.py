"""Portable, bounded telemetry collection inside the Ray driver and trainers.

This file and client.py are shipped as real Python modules with the Ray entrypoint.
Real files also allow Training Hub to serialize callback source for torchrun.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import socket
import subprocess
import threading
import time
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from stat import S_ISDIR
from typing import Any
from uuid import uuid4

try:
    from training_service_metrics_client import TrackingClient, TrackingError
except ImportError:
    from training_service.metrics.client import TrackingClient, TrackingError

LOG = logging.getLogger(__name__)
_CURRENT: RuntimeRecorder | None = None


def numeric_metrics(
    values: dict[str, Any], prefix: str = "", *, nonfinite: list[str] | None = None
) -> dict[str, float]:
    """Preserve backend numeric metrics; exclude text, nonfinite values and secrets."""
    result: dict[str, float] = {}

    def visit(value: Any, key: str, depth: int) -> None:
        if (
            len(result) >= 500
            or depth > 5
            or re.search(r"token(?!s)|password|secret|credential|api.key", key, re.I)
        ):
            return
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, key + "/" + str(name) if key else str(name), depth + 1)
        elif isinstance(value, Real) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                result[re.sub(r"[^\w /.:\-]", "_", key)[:250]] = number
            elif nonfinite is not None:
                nonfinite.append(key)

    visit(values, prefix.rstrip("/"), 0)
    return result


class SystemSampler:
    """Measure this pod's cgroup, not the entire host or an inferred cluster total."""

    root = Path("/sys/fs/cgroup")

    def sample(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for source, metric in (
            ("memory.current", "memory_bytes"),
            ("memory.peak", "memory_peak_bytes"),
        ):
            try:
                metrics[metric] = float((self.root / source).read_text().strip())
            except (OSError, ValueError):
                pass
        try:
            cpu = dict(line.split() for line in (self.root / "cpu.stat").read_text().splitlines())
            metrics["cpu_usage_seconds"] = float(cpu["usage_usec"]) / 1_000_000
            metrics["cpu_throttled_seconds"] = float(cpu.get("throttled_usec", 0)) / 1_000_000
        except (OSError, ValueError, KeyError):
            pass
        try:
            reads = writes = 0
            for line in (self.root / "io.stat").read_text().splitlines():
                values = dict(item.split("=", 1) for item in line.split()[1:])
                reads += int(values.get("rbytes", 0))
                writes += int(values.get("wbytes", 0))
            metrics.update(storage_read_bytes=float(reads), storage_write_bytes=float(writes))
        except (OSError, ValueError):
            pass
        return metrics

    def gpu(self) -> dict[str, float]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode:
                return {}
            metrics = {}
            for index, line in enumerate(result.stdout.splitlines()):
                for key, value, factor in zip(
                    (
                        "utilization_percent",
                        "memory_used_bytes",
                        "memory_total_bytes",
                        "power_watts",
                    ),
                    line.split(","),
                    (1, 2**20, 2**20, 1),
                    strict=True,
                ):
                    try:
                        metrics[f"gpu_{index}/{key}"] = float(value.strip()) * factor
                    except ValueError:
                        continue
            return metrics
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return {}


class RuntimeRecorder:
    """Network writes run on one bounded worker rather than blocking training hooks."""

    DRAIN_TIMEOUT = 10
    BATCH_SIZE = 64

    def __init__(self, configuration: dict[str, str]) -> None:
        self.configuration = configuration
        ca_bundle = "".join(
            os.environ[f"TRAINING_SERVICE_MLFLOW_CA_{index}"]
            for index in range(int(configuration.get("ca_bundle_parts", "0")))
        )
        self.client = TrackingClient(
            configuration["tracking_uri"],
            configuration["project"],
            lambda: Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip(),
            ca_bundle=ca_bundle,
        )
        self.run_id = configuration["run_id"]
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._failed = 0
        self._accepted = 0
        self._completed = 0
        self._nonfinite = 0
        self._lock = threading.Lock()
        self._closed = False
        identity = f"{socket.gethostname()}:{os.getpid()}"
        self._is_driver = os.getenv("TRAINING_SERVICE_MLFLOW_DRIVER_PROCESS", identity) == identity
        self._producer = "tracking.producer." + uuid4().hex
        self._rank = os.getenv("RANK")
        self._world_size = max(1, int(os.getenv("WORLD_SIZE", "1")))
        self._writer = threading.Thread(target=self._write, name="mlflow-metrics", daemon=True)
        self._sampler: threading.Thread | None = None
        self._step_started: float | None = None
        self._training_started: float | None = None
        self._checkpoint_count = 0
        self._initial_step: int | None = None
        self._last_step_duration: float | None = None
        self.emit(tags={self._producer: json.dumps(self._summary("running"))})
        self._writer.start()

    def emit(
        self,
        metrics: dict[str, Any] | None = None,
        *,
        tags: dict[str, str] | None = None,
        step: int = 0,
        directory_size: tuple[str, str] | None = None,
    ) -> None:
        omitted: list[str] = []
        values = numeric_metrics(metrics or {}, nonfinite=omitted)
        labels = dict(tags or {})
        if omitted:
            labels["training.nonfinite_metrics"] = "true"
        scan = None
        if directory_size:
            key, directory = directory_size
            try:
                scan = (key, directory, self._directory_identity(directory))
            except OSError:
                pass
        with self._lock:
            if self._closed:
                return
            self._nonfinite += len(omitted)
            try:
                self._queue.put_nowait(
                    (values, labels, max(0, int(step)), scan, int(time.time() * 1000))
                )
                self._accepted += 1
            except queue.Full:
                self._failed += 1

    @staticmethod
    def _directory_identity(directory: str) -> tuple[int, int, int, int]:
        stat = Path(directory).stat()
        if not S_ISDIR(stat.st_mode):
            raise NotADirectoryError(directory)
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns

    def _write(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                batch = [self._queue.get(timeout=0.1)]
            except queue.Empty:
                continue
            for _ in range(self.BATCH_SIZE - 1):
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                points: list[dict[str, Any]] = []
                labels = {}
                for metrics, tags, step, scan, timestamp in batch:
                    if scan:
                        key, directory, identity = scan
                        try:
                            if self._directory_identity(directory) == identity:
                                size = sum(
                                    p.stat().st_size
                                    for p in Path(directory).rglob("*")
                                    if p.is_file()
                                )
                                if self._directory_identity(directory) == identity:
                                    metrics[key] = float(size)
                        except OSError:
                            # Omit rotated/unavailable checkpoints rather than reporting zero.
                            pass
                    points.extend(
                        {"key": key, "value": value, "timestamp": timestamp, "step": step}
                        for key, value in metrics.items()
                    )
                    labels.update(tags)
                self.client.log_batch(self.run_id, points, labels)
            except TrackingError:
                with self._lock:
                    self._failed += len(batch)
                LOG.warning(
                    "MLflow metrics could not be delivered; check server access and readiness."
                )
            finally:
                with self._lock:
                    self._completed += len(batch)
                for _ in batch:
                    self._queue.task_done()

    def start_sampling(self) -> None:
        if self._sampler is not None:
            return

        def sample() -> None:
            sampler = SystemSampler()
            prefix = "system/pod_" + socket.gethostname() + "/"
            while not self._stop.is_set():
                self.emit({prefix + k: v for k, v in {**sampler.sample(), **sampler.gpu()}.items()})
                self._stop.wait(10)

        self._sampler = threading.Thread(target=sample, name="mlflow-resources", daemon=True)
        self._sampler.start()

    def event(self, event: str, context: Any) -> None:
        self._rank = os.getenv("RANK", "0")
        if not context.is_main_process:
            # Per-rank allocator counters are separate; never sum them as pod memory.
            if event in {"on_log", "on_train_end"}:
                self.process_memory(context.step)
            return
        now = time.time()
        metrics: dict[str, Any] = {
            "training/step": context.step,
            "training/epoch": context.epoch,
        }
        tags: dict[str, str] = {}
        directory_size = None
        if event == "on_train_begin":
            self._training_started = time.monotonic()
            self._initial_step = context.step
            metrics["training/started_at_unix_seconds"] = now
            metrics["training/initial_step"] = context.step
            tags["training.started_at"] = datetime.now(UTC).isoformat()
            tags["training.phase"] = "training"
            self.start_sampling()
        elif event == "on_step_begin":
            self._step_started = time.monotonic()
            return
        elif event == "on_step_end":
            if self._step_started is not None:
                self._last_step_duration = time.monotonic() - self._step_started
            # Avoid one HTTP write per optimizer step; on_log emits normal metrics.
            self._step_started = None
            return
        elif event in {"on_log", "on_evaluate"}:
            namespace = "eval" if event == "on_evaluate" else "train"
            metrics[namespace] = context.metrics
            if event == "on_log" and context.loss is not None:
                metrics[namespace + "/loss"] = context.loss
            elif event == "on_evaluate":
                # Some adapters carry the last training loss in context.loss during eval.
                for key in ("eval_loss", "val_loss"):
                    if key in (context.metrics or {}):
                        metrics["eval/loss"] = context.metrics[key]
                        break
            if context.learning_rate is not None:
                metrics["train/learning_rate"] = context.learning_rate
            if self._last_step_duration is not None:
                metrics["training/step_wall_seconds"] = self._last_step_duration
            self.process_memory(context.step)
        elif event == "on_save":
            self._checkpoint_count += 1
            metrics.update(
                {
                    "checkpoint/saved_at_unix_seconds": now,
                    "checkpoint/count": self._checkpoint_count,
                    "checkpoint/success": 1,
                }
            )
            raw = context.metrics or {}
            path = raw.get("checkpoint_path")
            size = raw.get("checkpoint_size_bytes")
            if (
                isinstance(size, Real)
                and not isinstance(size, bool)
                and math.isfinite(size)
                and float(size) >= 0
            ):
                metrics["checkpoint/size_bytes"] = size
            if isinstance(path, str):
                tags["checkpoint.latest_path"] = path[:1000]
                if "checkpoint/size_bytes" not in metrics:
                    directory_size = ("checkpoint/size_bytes", path)
            # Do not call time since last step 'checkpoint duration': it includes other work.
            for key in ("checkpoint_duration_seconds", "checkpoint_save_duration_seconds"):
                if isinstance(raw.get(key), Real):
                    metrics["checkpoint/duration_seconds"] = raw[key]
            tags["checkpoint.duration_source"] = (
                "backend" if "checkpoint/duration_seconds" in metrics else "not_exposed"
            )
            if isinstance(raw.get("checkpoint_type"), str):
                tags["checkpoint.latest_type"] = raw["checkpoint_type"][:100]
        elif event == "on_train_end":
            metrics["training/finished_at_unix_seconds"] = now
            if self._training_started is not None:
                metrics["training/duration_seconds"] = time.monotonic() - self._training_started
            tags["training.finished_at"] = datetime.now(UTC).isoformat()
        self.emit(metrics, tags=tags, step=context.step, directory_size=directory_size)

    def process_memory(self, step: int) -> None:
        import resource
        import sys

        rank = os.getenv("RANK", "0")
        prefix = f"process/rank_{rank}/"
        usage = resource.getrusage(resource.RUSAGE_SELF)
        metrics = {
            prefix + "cpu_user_seconds": usage.ru_utime,
            prefix + "cpu_system_seconds": usage.ru_stime,
            prefix + "max_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        }
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            metrics[prefix + "cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            metrics[prefix + "cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        self.emit(metrics, step=step)

    def _summary(self, state: str | None = None) -> dict[str, Any]:
        with self._lock:
            unresolved = self._accepted - self._completed
            return {
                "state": state or ("incomplete" if self._failed or unresolved else "complete"),
                "failed": self._failed,
                "unresolved": unresolved,
                "nonfinite": self._nonfinite,
                "rank": self._rank,
                "world_size": self._world_size,
            }

    def _aggregate_delivery(self, own: dict[str, Any]) -> None:
        run = self.client.get_run(self.run_id)
        tags = {tag["key"]: tag["value"] for tag in run.get("data", {}).get("tags", [])}
        summaries = {}
        try:
            for key, value in tags.items():
                if key.startswith("tracking.producer."):
                    item = json.loads(value)
                    if not isinstance(item, dict) or any(
                        not isinstance(item.get(field), int) or item[field] < 0
                        for field in ("failed", "unresolved", "nonfinite", "world_size")
                    ):
                        raise ValueError("Invalid producer summary")
                    summaries[key] = item
        except (ValueError, TypeError):
            raise TrackingError() from None
        summaries[self._producer] = own
        complete = all(item.get("state") == "complete" for item in summaries.values())
        if tags.get("tracking.unified_callbacks") == "true":
            expected = max(
                int(self.configuration.get("expected_ranks", "1")),
                *(item["world_size"] for item in summaries.values()),
            )
            ranks = {
                item.get("rank") for item in summaries.values() if item.get("state") == "complete"
            }
            complete = complete and all(str(rank) in ranks for rank in range(expected))
        failed = sum(item["failed"] for item in summaries.values())
        unresolved = sum(item["unresolved"] for item in summaries.values())
        self.client.log(
            self.run_id,
            metrics={
                "tracking/undelivered_events": failed + unresolved,
                "tracking/failed_events": failed,
                "tracking/unresolved_events": unresolved,
                "tracking/nonfinite_values": sum(item["nonfinite"] for item in summaries.values()),
            },
            tags={
                "tracking.delivery": "complete" if complete else "incomplete",
                **(
                    {"training.nonfinite_metrics": "true"}
                    if any(item["nonfinite"] for item in summaries.values())
                    else {}
                ),
            },
        )

    def close(self, status: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=3)
        self._writer.join(timeout=self.DRAIN_TIMEOUT)
        summary = self._summary()
        try:
            self.client.log(self.run_id, tags={self._producer: json.dumps(summary)})
        except TrackingError:
            LOG.warning("MLflow delivery summary could not be saved; check tracking availability.")
        if self._is_driver:
            try:
                self._aggregate_delivery(summary)
            except TrackingError:
                LOG.warning(
                    "MLflow delivery health could not be verified; inspect producer summaries."
                )
        # Ancillary telemetry must never suppress an independent terminal-state attempt.
        if status:
            try:
                self.client.finish(self.run_id, status)
            except TrackingError:
                LOG.warning("MLflow finalization failed; the run may remain open in MLflow.")


def current_recorder() -> RuntimeRecorder:
    global _CURRENT
    if _CURRENT is None:
        import atexit

        _CURRENT = RuntimeRecorder(json.loads(os.environ["TRAINING_SERVICE_MLFLOW_RUN"]))
        atexit.register(_CURRENT.close)
    return _CURRENT
