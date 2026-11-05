"""Focused proof that the generated contract is wired into the application."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from training_service.contract import load_contract
from training_service.main import app
from training_service_api.models.create_job_request import CreateJobRequest

client = TestClient(app)


def test_startup_and_docs() -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    document = client.get("/openapi.json").json()
    assert document["servers"] == [{"url": "/api/v1"}]


def test_all_generated_routes_fail_closed_without_backend_implementations() -> None:
    examples = load_contract()["paths"]["/projects/{project}/training-jobs"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]
    job = next(iter(examples.values()))["value"]
    estimate = {
        "model": {"source": "huggingface", "uri": "meta-llama/Llama-3.2-3B"},
        "dataset": {"uri": "/data/train.jsonl"},
        "algorithm": {"name": "lora_sft", "parameters": {"learning_rate": 0.00002}},
    }
    cases = (
        ("get", "/algorithms", None),
        ("get", "/algorithms/example", None),
        ("get", "/projects/example/queues", None),
        ("post", "/training-jobs/estimate", estimate),
        ("get", "/projects/example/training-jobs", None),
        ("post", "/projects/example/training-jobs", job),
        ("get", "/projects/example/training-jobs/example", None),
        ("delete", "/projects/example/training-jobs/example", None),
        ("get", "/projects/example/training-jobs/example/logs", None),
    )
    expected = {
        (
            method,
            path.replace("{project}", "example")
            .replace("{job_id}", "example")
            .replace("{name}", "example"),
        )
        for path, item in load_contract()["paths"].items()
        for method, operation in item.items()
        if isinstance(operation, dict) and operation.get("operationId")
    }
    assert len(expected) == 9
    assert {(method, path) for method, path, _ in cases} == expected
    for method, path, body in cases:
        response = client.request(method, "/api/v1" + path, json=body)
        assert response.status_code == 401, response.text
        assert response.json()["code"] == "unauthorized"


def test_generated_models_preserve_approved_wire_format() -> None:
    examples = load_contract()["paths"]["/projects/{project}/training-jobs"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]
    for entry in examples.values():
        payload = entry["value"]
        assert (
            CreateJobRequest.model_validate(payload).model_dump(mode="json", exclude_unset=True)
            == payload
        )
    valid = next(iter(examples.values()))["value"]
    for field, value in (
        ("dataset", {"config": ""}),
        ("output", {"secret_ref": "obsolete-reference"}),
        ("dataset", {"uri": "pvc://training-data/../escape"}),
    ):
        invalid = deepcopy(valid)
        invalid[field].update(value)
        with pytest.raises(ValidationError):
            CreateJobRequest.model_validate(invalid)
