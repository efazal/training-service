"""Read-only discovery checks over the live deployed HTTP API."""

import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from training_service.contract import component_validator

BASE = "/api/v1/algorithms"


def test_compact_catalog_matches_configured_runtime(api):
    data = api.object(api.request("GET", BASE))
    assert isinstance(data["algorithms"], list)
    for algorithm in data["algorithms"]:
        component_validator("AlgorithmDescriptor").validate(algorithm)
        assert "parameter_schema" not in algorithm
        assert "Args:" not in algorithm["description"]
        assert "Examples:" not in algorithm["description"]
        assert algorithm["details_url"] == BASE + "/" + algorithm["name"]
    assert data.get("runtime", {}).get("training_hub_version")
    if image := os.getenv("TRAINING_SERVICE_RUNTIME_IMAGE"):
        assert data["runtime"]["image"] == image
    if directory := os.getenv("TRAINING_SERVICE_EVIDENCE_DIR"):
        (Path(directory) / "algorithm-discovery.json").write_text(json.dumps(data, indent=2))


def test_follow_algorithm_details_and_opt_into_schema(api):
    summaries = api.object(api.request("GET", BASE))["algorithms"]
    for summary in summaries:
        detail = api.object(api.request("GET", summary["details_url"]))
        component_validator("AlgorithmDetails").validate(detail)
        assert "parameter_schema" not in detail
        assert "example_request" not in detail
        assert {key: detail[key] for key in summary} == summary
        expanded = api.object(api.request("GET", detail["parameter_schema_url"]))
        schema = expanded.pop("parameter_schema")
        Draft202012Validator.check_schema(schema)
        assert expanded == detail
        assert set(detail["parameters"]) == set(schema.get("properties", {}))
        assert detail["required_parameters"] == schema.get("required", [])
        for name, parameter in detail["parameters"].items():
            if "example" in parameter:
                native = schema["properties"][name]
                assert "example" in native
                assert parameter["example"] == native["example"]
                Draft202012Validator(native).validate(parameter["example"])


def test_missing_algorithm_has_context(api, missing_job_id):
    api.error(
        api.request("GET", BASE + "/" + missing_job_id, expected=404),
        "not_found",
        mentions=(missing_job_id, "catalog"),
    )


@pytest.mark.parametrize("include", ["all", "", "parameter_schema,other"])
def test_invalid_discovery_query(api, include):
    api.error(
        api.request("GET", BASE + "/lora_sft", params={"include": include}, expected=400),
        "bad_request",
    )


@pytest.mark.parametrize("suffix", ["", "/lora_sft", "/lora_sft?include=parameter_schema"])
@pytest.mark.parametrize("invalid_token", [False, True])
def test_discovery_requires_valid_token(api, suffix, invalid_token):
    options = (
        {"headers": {"Authorization": "Bearer e2e-invalid-token"}}
        if invalid_token
        else {"authenticated": False}
    )
    response = api.request("GET", BASE + suffix, expected=401, **options)
    api.error(response, "unauthorized")
    assert response.headers.get("www-authenticate", "").lower() == "bearer"


@pytest.mark.parametrize("failure", ["algorithm", "parameter", "type"])
def test_submission_rejects_runtime_incompatibility_before_storage_lookup(
    api, missing_job_id, failure
):
    summaries = api.object(api.request("GET", BASE))["algorithms"]
    assert summaries, "Runtime must expose algorithms for training"
    name = next((a["name"] for a in summaries if a["name"] == "lora_sft"), summaries[0]["name"])
    parameters = {}
    if failure == "algorithm":
        name = missing_job_id
        assert name not in {a["name"] for a in summaries}
    elif failure == "parameter":
        parameters["unsupported_e2e_parameter"] = 1
    else:
        parameters["learning_rate"] = {"invalid": "not a number"}
    payload = {
        "name": missing_job_id,
        "model": {"uri": f"pvc://{missing_job_id}/model/"},
        "dataset": {"uri": f"pvc://{missing_job_id}/train.jsonl"},
        "algorithm": {"name": name, "parameters": parameters},
    }
    # Missing PVCs would return 404: 422 proves validation happened first.
    response = api.request("POST", api.jobs, expected=422, json=payload)
    api.error(response, "validation_error", mentions=(name,))


def test_submission_reports_missing_runtime_required_parameters(api, missing_job_id):
    summaries = api.object(api.request("GET", BASE))["algorithms"]
    checked = 0
    for summary in summaries:
        detail = api.object(api.request("GET", summary["details_url"]))
        if not detail["required_parameters"]:
            continue
        payload = {
            "name": missing_job_id,
            "model": {"uri": f"pvc://{missing_job_id}/model/"},
            "dataset": {"uri": f"pvc://{missing_job_id}/train.jsonl"},
            "algorithm": {"name": summary["name"], "parameters": {}},
        }
        response = api.request("POST", api.jobs, expected=422, json=payload)
        api.error(response, "validation_error", mentions=(summary["name"],))
        checked += 1
    if not checked:
        pytest.skip("Selected runtime has no algorithms with required JSON parameters")
