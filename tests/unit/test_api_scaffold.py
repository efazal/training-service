"""Focused proof that the generated contract is wired into the application."""

import tomllib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from training_service import __version__
from training_service.contract import load_contract
from training_service.main import app
from training_service_api.models.create_job_request import CreateJobRequest

client = TestClient(app)


def test_startup_and_docs() -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    document = client.get("/openapi.json").json()
    canonical = deepcopy(load_contract())
    canonical["servers"] = [{"url": "/api/v1"}]
    assert document == canonical


def test_public_api_version_matches_release_metadata() -> None:
    root = Path(__file__).parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    chart = yaml.safe_load((root / "charts/training-service/Chart.yaml").read_text())
    values = yaml.safe_load((root / "charts/training-service/values.yaml").read_text())
    generator = yaml.safe_load((root / "docs/api/openapi-generator-config.yaml").read_text())
    versions = {
        "openapi": client.get("/openapi.json").json()["info"]["version"],
        "package": project["project"]["version"],
        "chart_app": chart["appVersion"],
        "image": values["image"]["tag"],
        "generated_package": generator["packageVersion"],
    }
    assert versions == dict.fromkeys(versions, __version__)


def test_estimation_is_absent_from_contract_and_routes() -> None:
    document = client.get("/openapi.json").json()
    assert "/training-jobs/estimate" not in document["paths"]
    assert not any("estimation" in name.lower() for name in document["components"]["schemas"])
    assert client.post("/api/v1/training-jobs/estimate", json={}).status_code == 404


def test_all_generated_routes_fail_closed_without_backend_implementations() -> None:
    examples = load_contract()["paths"]["/projects/{project}/training-jobs"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]
    job = next(iter(examples.values()))["value"]
    cases = (
        ("get", "/algorithms", None),
        ("get", "/algorithms/example", None),
        ("get", "/projects/example/queues", None),
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
    assert len(expected) == 8
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
