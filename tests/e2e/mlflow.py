"""Independent MLflow oracle; never send a token to a job-provided URL."""

import math
import os
from urllib.parse import urlencode

from training_service.metrics.client import TrackingClient, TrackingError


def verify_run(api, uri, detail):
    tracking = TrackingClient(
        uri,
        api.project,
        lambda: api.client.headers["Authorization"].removeprefix("Bearer "),
        ca_file=os.getenv("TRAINING_SERVICE_CA_BUNDLE") or None,
    )
    run_id = detail.get("mlflow_run_id")
    assert isinstance(run_id, str), "Job did not expose an MLflow run ID"
    try:
        run = tracking.request("GET", "runs/get", {"run_id": run_id})["run"]
        info = run["info"]
        expected = (
            uri.rstrip("/")
            + "?"
            + urlencode({"workspace": api.project})
            + "#/experiments/"
            + str(info["experiment_id"])
            + "/runs/"
            + run_id
        )
        assert detail.get("mlflow_url") == expected, (
            "Job link does not match the trusted MLflow server and project"
        )
        assert info["status"] == "FINISHED", "MLflow did not record training completion"
        tags = {t["key"]: t["value"] for t in run["data"]["tags"]}
        assert tags.get("training.project") == api.project
        assert tags.get("training.job_id") == detail["id"]
        assert tags.get("tracking.unified_callbacks") == "true", (
            "Detailed metrics require a callback-enabled runtime and algorithm"
        )
        values = {m["key"]: m["value"] for m in run["data"]["metrics"]}
        for key in ("train/loss", "training/epoch", "training/step", "job/duration_seconds"):
            assert key in values and math.isfinite(values[key]), (
                f"Missing numeric MLflow metric: {key}"
            )
        assert (
            values["training/finished_at_unix_seconds"]
            >= values["training/started_at_unix_seconds"]
        )
        history = tracking.request(
            "GET", "metrics/get-history", {"run_id": run_id, "metric_key": "train/loss"}
        )["metrics"]
        assert history and all(math.isfinite(m["value"]) and m["step"] >= 0 for m in history)
        assert tags.get("tracking.delivery") == "complete", "Metric delivery was incomplete"
    except TrackingError:
        raise AssertionError(
            "Cannot verify MLflow run with the caller's project access and trusted CA"
        ) from None
    return {
        "run_id": run_id,
        "experiment_id": info["experiment_id"],
        "loss_points": len(history),
        "persisted": True,
    }
