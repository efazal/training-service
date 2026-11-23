"""Adversarial checks for the live metrics oracle, without a running training job."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.e2e import mlflow

from training_service.metrics.client import TrackingClient

URI = "https://tracking.example/mlflow"
RUN = "a" * 32


@pytest.fixture
def oracle(monkeypatch):
    run = {
        "info": {"run_id": RUN, "experiment_id": "42", "status": "FINISHED"},
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
    # Exercise the real get_run validation; mock only the HTTP boundary.
    tracking_client = TrackingClient(URI, "team-alpha", lambda: "caller-credential")

    def request(method, endpoint, data):
        assert method == "GET"
        if endpoint == "runs/get":
            assert data == {"run_id": RUN}
            return {"run": run}
        assert endpoint == "metrics/get-history"
        page_size = data["max_results"]
        assert isinstance(page_size, int) and 0 < page_size <= 1000
        expected = {"run_id": RUN, "metric_key": "train/loss", "max_results": page_size}
        offset = 0
        if "page_token" in data:
            expected["page_token"] = data["page_token"]
            offset = int(data["page_token"])
        assert data == expected
        response = {"metrics": history[offset : offset + page_size]}
        if offset + page_size < len(history):
            response["next_page_token"] = str(offset + page_size)
        return response

    monkeypatch.setattr(tracking_client, "request", Mock(side_effect=request))
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
        "lifecycle_metrics": {
            "job/duration_seconds": 10,
            "job/started_at_unix_seconds": 100,
            "job/finished_at_unix_seconds": 110,
            "training/started_at_unix_seconds": 100,
            "training/finished_at_unix_seconds": 110,
        },
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


def test_metrics_oracle_rejects_mismatched_returned_run_id(oracle):
    api, detail, run, _, _ = oracle
    run["info"]["run_id"] = "b" * 32
    with pytest.raises(AssertionError, match="Cannot verify MLflow run"):
        mlflow.verify_run(api, URI, detail, detailed=False)


def test_loss_history_paginates_all_points_and_preserves_timestamps(oracle):
    api, detail, _, history, constructor = oracle
    history[:] = [
        {"value": 1 / (step + 1), "step": step, "timestamp": 105000 + step} for step in range(2005)
    ]
    before = mlflow.verify_run(api, URI, detail)
    assert before["loss_points"] == len(history)
    assert (
        before["loss_history_sha256"]
        == hashlib.sha256(
            json.dumps(sorted((m["step"], m["timestamp"], m["value"]) for m in history)).encode()
        ).hexdigest()
    )
    requests = constructor.return_value.request.call_args_list
    pages = [call.args[2] for call in requests if call.args[1] == "metrics/get-history"]
    assert len(pages) >= 3
    assert "page_token" not in pages[0]
    assert pages[1]["page_token"] == str(pages[0]["max_results"])
    history.reverse()
    assert mlflow.verify_run(api, URI, detail) == before
    history[-1]["timestamp"] += 1
    assert mlflow.verify_run(api, URI, detail) != before


@pytest.mark.parametrize("tokens", [["repeat", "repeat"], ["first", "second", "first"]])
def test_loss_history_rejects_repeated_or_cyclic_tokens(oracle, tokens):
    api, detail, run, history, constructor = oracle
    request = constructor.return_value.request
    request.side_effect = [{"run": run}] + [
        {"metrics": history, "next_page_token": token} for token in tokens
    ]
    with pytest.raises(AssertionError, match="pagination token"):
        mlflow.verify_run(api, URI, detail)
    assert request.call_count == 1 + len(tokens)


@pytest.mark.parametrize(
    "page",
    [
        {"metrics": {}},
        {"metrics": None},
        {"metrics": [None] * 1001},
        {"metrics": [], "next_page_token": 42},
        {"metrics": [], "next_page_token": []},
    ],
)
def test_loss_history_rejects_malformed_or_oversized_pages(oracle, page):
    api, detail, run, _, constructor = oracle
    request = constructor.return_value.request
    request.side_effect = [{"run": run}, page]
    with pytest.raises(AssertionError, match="metric history|pagination token"):
        mlflow.verify_run(api, URI, detail)
    assert request.call_count == 2


def test_loss_history_bounds_even_unique_continuation_tokens(oracle, monkeypatch):
    api, detail, run, history, constructor = oracle
    monkeypatch.setattr(mlflow, "_HISTORY_MAX_PAGES", 3)
    request = constructor.return_value.request
    request.side_effect = [{"run": run}] + [
        {"metrics": history, "next_page_token": str(index)} for index in range(3)
    ]
    with pytest.raises(AssertionError, match="pagination limit"):
        mlflow.verify_run(api, URI, detail)
    assert request.call_count == 4


@pytest.mark.parametrize("token", [None, ""])
def test_loss_history_accepts_explicit_end_of_pagination(oracle, token):
    api, detail, run, history, constructor = oracle
    constructor.return_value.request.side_effect = [
        {"run": run},
        {"metrics": history, "next_page_token": token},
    ]
    assert mlflow.verify_run(api, URI, detail)["loss_points"] == len(history)


def test_loss_history_follows_opaque_token_after_empty_page(oracle):
    api, detail, run, history, constructor = oracle
    request = constructor.return_value.request
    request.side_effect = [
        {"run": run},
        {"next_page_token": "opaque+/=token"},
        {"metrics": history},
    ]
    assert mlflow.verify_run(api, URI, detail)["loss_points"] == len(history)
    assert request.call_args.args == (
        "GET",
        "metrics/get-history",
        {
            "run_id": RUN,
            "metric_key": "train/loss",
            "max_results": 1000,
            "page_token": "opaque+/=token",
        },
    )


@pytest.mark.parametrize("detailed", [False, True])
def test_persistence_detects_changed_lifecycle_metrics(oracle, detailed):
    api, detail, run, _, _ = oracle
    before = mlflow.verify_run(api, URI, detail, detailed=detailed)
    for metric in run["data"]["metrics"]:
        if metric["key"] == "job/duration_seconds":
            metric["value"] = 12
    assert mlflow.verify_run(api, URI, detail, detailed=detailed) != before


def test_generic_profiles_verify_lifecycle_without_detailed_callbacks(oracle):
    api, detail, run, _, _ = oracle
    run["data"]["tags"] = [
        t for t in run["data"]["tags"] if t["key"] != "tracking.unified_callbacks"
    ]
    run["data"]["metrics"] = [m for m in run["data"]["metrics"] if m["key"].startswith("job/")]
    assert mlflow.verify_run(api, URI, detail, detailed=False)["persisted"]
