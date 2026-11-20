"""Adversarial checks for the live metrics oracle, without a running training job."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.e2e import mlflow

URI = "https://tracking.example/mlflow"
RUN = "a" * 32


@pytest.fixture
def oracle(monkeypatch):
    run = {
        "info": {"experiment_id": "42", "status": "FINISHED"},
        "data": {
            "tags": [
                {"key": key, "value": value}
                for key, value in {
                    "training.project": "team-alpha",
                    "training.job_id": "job-1",
                    "tracking.unified_callbacks": "true",
                    "tracking.delivery": "complete",
                }.items()
            ],
            "metrics": [
                {"key": key, "value": value}
                for key, value in {
                    "train/loss": 0.5,
                    "training/epoch": 1,
                    "training/step": 2,
                    "job/duration_seconds": 10,
                    "job/started_at_unix_seconds": 100,
                    "job/finished_at_unix_seconds": 110,
                    "training/started_at_unix_seconds": 100,
                    "training/finished_at_unix_seconds": 110,
                }.items()
            ],
        },
    }
    history = [{"value": 0.5, "step": 2, "timestamp": 105000}]
    tracking_client = Mock()

    def request(method, endpoint, data):
        assert method == "GET"
        if endpoint == "runs/get":
            assert data == {"run_id": RUN}
            return {"run": run}
        assert endpoint == "metrics/get-history"
        assert data == {"run_id": RUN, "metric_key": "train/loss"}
        return {"metrics": history}

    tracking_client.request.side_effect = request
    constructor = Mock(return_value=tracking_client)
    monkeypatch.setattr(mlflow, "TrackingClient", constructor)
    api = SimpleNamespace(
        project="team-alpha",
        client=SimpleNamespace(headers={"Authorization": "Bearer caller-credential"}),
    )
    detail = {
        "id": "job-1",
        "mlflow_run_id": RUN,
        "mlflow_url": URI + "?workspace=team-alpha#/experiments/42/runs/" + RUN,
    }
    return api, detail, run, history, constructor


def test_metrics_oracle_requires_actual_history_and_can_verify_after_job_deletion(oracle):
    api, detail, _, _, constructor = oracle
    expected = {
        "run_id": RUN,
        "experiment_id": "42",
        "loss_points": 1,
        "persisted": True,
        "loss_history_sha256": hashlib.sha256(json.dumps([(2, 105000, 0.5)]).encode()).hexdigest(),
    }
    assert mlflow.verify_run(api, URI, detail) == expected
    assert mlflow.verify_run(api, URI, deepcopy(detail)) == expected
    assert all(call.args[:2] == (URI, "team-alpha") for call in constructor.call_args_list)


@pytest.mark.parametrize(
    "fault",
    [
        "no_loss",
        "failed",
        "wrong_project",
        "no_callbacks",
        "incomplete",
        "empty_history",
        "nonfinite_loss",
        "untrusted_link",
    ],
)
def test_metrics_oracle_rejects_missing_or_misattributed_metrics(oracle, fault):
    api, detail, run, history, constructor = oracle
    if fault == "no_loss":
        run["data"]["metrics"] = [m for m in run["data"]["metrics"] if m["key"] != "train/loss"]
    elif fault == "failed":
        run["info"]["status"] = "FAILED"
    elif fault in {"wrong_project", "no_callbacks", "incomplete"}:
        key, value = {
            "wrong_project": ("training.project", "other-project"),
            "no_callbacks": ("tracking.unified_callbacks", "false"),
            "incomplete": ("tracking.delivery", "incomplete"),
        }[fault]
        for tag in run["data"]["tags"]:
            if tag["key"] == key:
                tag["value"] = value
    elif fault == "empty_history":
        history.clear()
    elif fault == "nonfinite_loss":
        history[0]["value"] = float("nan")
    elif fault == "untrusted_link":
        detail["mlflow_url"] = "https://untrusted.example/collect"
    with pytest.raises(AssertionError):
        mlflow.verify_run(api, URI, detail)
    assert constructor.call_args.args[0] == URI


def test_persistence_detects_changed_points_with_same_count(oracle):
    api, detail, _, history, _ = oracle
    before = mlflow.verify_run(api, URI, detail)
    history[0].update(step=3, value=0.9)
    assert mlflow.verify_run(api, URI, detail) != before


def test_persistence_is_independent_of_response_order(oracle):
    api, detail, _, history, _ = oracle
    history.append({"value": 0.4, "step": 3, "timestamp": 106000})
    before = mlflow.verify_run(api, URI, detail)
    history.reverse()
    assert mlflow.verify_run(api, URI, detail) == before


def test_generic_profiles_verify_lifecycle_without_detailed_callbacks(oracle):
    api, detail, run, _, _ = oracle
    run["data"]["tags"] = [
        t for t in run["data"]["tags"] if t["key"] != "tracking.unified_callbacks"
    ]
    run["data"]["metrics"] = [m for m in run["data"]["metrics"] if m["key"].startswith("job/")]
    assert mlflow.verify_run(api, URI, detail, detailed=False)["persisted"]
