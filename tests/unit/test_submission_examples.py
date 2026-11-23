"""Copy discovery examples into the real submission route and its validators."""

import json
from copy import deepcopy
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from kubernetes.client.exceptions import ApiException
from referencing import Registry

from training_service import jobs
from training_service.auth import CallerIdentity
from training_service.contract import component_validator
from training_service.main import create_app
from training_service_api.models.create_job_request import CreateJobRequest

HEADERS = {"Authorization": "Bearer test-only-token"}
ALGORITHMS = ("sft", "osft", "lora_sft", "lora_grpo")


class Authenticator:
    async def authenticate(self, token):
        return CallerIdentity("alice", "uid", ())


class Authorizer:
    async def is_allowed(self, identity, project, permission):
        return True


@pytest.fixture
def client():
    return TestClient(
        create_app(
            Authenticator(),
            Authorizer(),
            backend=jobs.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    )


@pytest.mark.parametrize("name", ALGORITHMS)
def test_discovered_example_passes_post_validation_and_reaches_storage(client, monkeypatch, name):
    core = Mock()
    core.read_namespaced_persistent_volume_claim.side_effect = ApiException(status=404)
    monkeypatch.setattr(jobs, "_caller_api_client", lambda token: MagicMock())
    monkeypatch.setattr(jobs.client, "CoreV1Api", lambda api: core)
    monkeypatch.setattr(jobs, "resolve_local_queue", lambda api, project, queue: None)

    response = client.get(f"/api/v1/algorithms/{name}?include=parameter_schema", headers=HEADERS)
    assert response.status_code == 200
    detail = response.json()
    example = detail["example_request"]
    component_validator("CreateJobRequest").validate(example)
    assert CreateJobRequest.model_validate(example).to_dict() == example
    assert example["algorithm"]["name"] == name
    Draft202012Validator(detail["parameter_schema"], registry=Registry()).validate(
        example["algorithm"]["parameters"]
    )
    assert example["resources"]["num_nodes"] == (2 if name == "lora_grpo" else 1)
    assert example["resources"]["gpus_per_node"] == 1
    assert example["output"]["destination_uri"].startswith("pvc://training-data/")

    # No edits to the discovered body: missing project storage is the first failure.
    result = client.post("/api/v1/projects/team-alpha/training-jobs", headers=HEADERS, json=example)
    assert result.status_code == 404, result.text
    assert result.json()["code"] == "not_found"
    assert "training-data" in result.json()["message"]
    assert core.read_namespaced_persistent_volume_claim.call_count == 1


@pytest.mark.parametrize("name", ALGORITHMS)
def test_examples_are_in_regular_details_and_do_not_mutate_catalog(client, runtime_catalog, name):
    original = runtime_catalog.read_text()
    plain = client.get(f"/api/v1/algorithms/{name}", headers=HEADERS).json()
    expanded = client.get(plain["parameter_schema_url"], headers=HEADERS).json()
    assert plain["example_request"] == expanded["example_request"]
    assert any("example" in note.lower() and "storage" in note.lower() for note in plain["notes"])
    assert runtime_catalog.read_text() == original


def test_unsupported_execution_profile_has_no_submission_example(client):
    result = client.get("/api/v1/algorithms/grpo", headers=HEADERS)
    assert result.status_code == 200
    assert "example_request" not in result.json()


def test_new_required_parameter_uses_runtime_example(client, runtime_catalog):
    payload = json.loads(runtime_catalog.read_text())
    schema = next(a for a in payload["algorithms"] if a["name"] == "sft")["parameter_schema"]
    schema["properties"]["new_required_argument"] = {
        "type": "string",
        "enum": ["supported-mode"],
        "example": "supported-mode",
    }
    schema["required"].append("new_required_argument")
    runtime_catalog.write_text(json.dumps(payload))
    result = client.get("/api/v1/algorithms/sft", headers=HEADERS)
    assert result.status_code == 200
    example = result.json()["example_request"]
    assert example["algorithm"]["parameters"]["new_required_argument"] == "supported-mode"
    Draft202012Validator(schema, registry=Registry()).validate(example["algorithm"]["parameters"])


def test_runtime_changes_cannot_leave_invalid_recipe_parameters(client, runtime_catalog):
    payload = json.loads(runtime_catalog.read_text())
    schema = next(a for a in payload["algorithms"] if a["name"] == "lora_sft")["parameter_schema"]
    del schema["properties"]["num_epochs"]
    schema["properties"]["learning_rate"] = {"type": "number", "const": 0.001}
    runtime_catalog.write_text(json.dumps(payload))
    detail = client.get("/api/v1/algorithms/lora_sft", headers=HEADERS).json()
    example = detail["example_request"]["algorithm"]["parameters"]
    assert "num_epochs" not in example
    if "learning_rate" in example:
        assert example["learning_rate"] == 0.001
    Draft202012Validator(schema, registry=Registry()).validate(example)


def test_runtime_does_not_supply_example_for_python_only_profile(client, runtime_catalog):
    payload = json.loads(runtime_catalog.read_text())
    schema = next(a for a in payload["algorithms"] if a["name"] == "lora_sft")["parameter_schema"]
    schema["x-training-hub"]["api_callable"] = False
    runtime_catalog.write_text(json.dumps(payload))
    result = client.get("/api/v1/algorithms/lora_sft", headers=HEADERS)
    assert result.status_code == 200
    assert "example_request" not in result.json()


def test_mutating_returned_example_does_not_affect_later_requests(client):
    first = client.get("/api/v1/algorithms/lora_grpo", headers=HEADERS).json()["example_request"]
    original = deepcopy(first)
    first["resources"]["num_nodes"] = 1
    first["algorithm"]["parameters"].clear()
    second = client.get("/api/v1/algorithms/lora_grpo", headers=HEADERS).json()["example_request"]
    assert second == original
