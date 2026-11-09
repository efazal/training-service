"""Read-only discovery checks over the live deployed HTTP API."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry

from tests.e2e.algorithms import ALGORITHMS, PARAMETERS
from training_service.contract import component_validator

BASE = "/api/v1/algorithms"


@pytest.fixture(params=ALGORITHMS)
def runtime_algorithm(api, request):
    name = request.param
    detail = api.object(
        api.request("GET", f"{BASE}/{name}", params={"include": "parameter_schema"})
    )
    schema = detail["parameter_schema"]
    parameters = deepcopy(PARAMETERS[name])
    if name == "lora_grpo":
        # The separate math training matrix needs reward_type, which stock
        # Training Hub 0.11.2 does not expose. Exercise its actual JSON API here.
        parameters = {"num_iterations": 1, "group_size": 4, "prompt_batch_size": 4}
    # The HTTP-advertised schema is an independent oracle, not the server validator.
    Draft202012Validator(schema, registry=Registry()).validate(parameters)
    return name, schema, parameters


@pytest.fixture
def catalog_submission(api, missing_job_id, pytestconfig):
    payloads = []

    def submit(name, parameters, expected, mentions=()):
        payload = {
            "name": f"e2e-{missing_job_id[-16:]}-{len(payloads)}",
            "model": {"uri": f"pvc://{missing_job_id}/model/"},
            "dataset": {"uri": f"pvc://{missing_job_id}/train.jsonl"},
            "output": {"destination_uri": f"pvc://{missing_job_id}/output/"},
            "algorithm": {"name": name, "parameters": parameters},
            "resources": {"num_nodes": 2 if name == "lora_grpo" else 1, "gpus_per_node": 1},
        }
        payloads.append(payload)
        response = api.request("POST", api.jobs, expected=expected, json=payload)
        api.error(
            response,
            "validation_error" if expected == 422 else "not_found",
            mentions=mentions,
        )
        api.error(api.request("GET", f"{api.jobs}/{payload['name']}", expected=404), "not_found")
        return api.object(response)

    try:
        yield submit
    finally:
        for payload in payloads:
            api.cleanup(payload["name"], pytestconfig.getoption("cleanup_timeout"), payload)


def test_compact_catalog_matches_configured_runtime(api):
    data = api.object(api.request("GET", BASE))
    assert isinstance(data["algorithms"], list)
    names = [algorithm["name"] for algorithm in data["algorithms"]]
    assert len(names) == len(set(names)), "Algorithm names must be unique"
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
    api, missing_job_id, catalog_submission, failure
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
    # Missing PVCs would return 404: 422 proves validation happened first.
    catalog_submission(name, parameters, 422, (name,))


def test_submission_reports_missing_runtime_required_parameters(api, catalog_submission):
    summaries = api.object(api.request("GET", BASE))["algorithms"]
    checked = 0
    for summary in summaries:
        detail = api.object(api.request("GET", summary["details_url"]))
        if not detail["required_parameters"]:
            continue
        catalog_submission(
            summary["name"],
            {},
            422,
            (
                summary["name"],
                "Missing required parameter(s)",
                *detail["required_parameters"],
            ),
        )
        checked += 1
    if not checked:
        pytest.skip("Selected runtime has no algorithms with required JSON parameters")


@pytest.mark.parametrize("defaults_only", [False, True], ids=["representative", "defaults"])
def test_valid_runtime_parameters_reach_storage_lookup(
    runtime_algorithm, catalog_submission, missing_job_id, defaults_only
):
    name, schema, parameters = runtime_algorithm
    if defaults_only:
        parameters = {key: parameters[key] for key in schema.get("required", [])}
    Draft202012Validator(schema, registry=Registry()).validate(parameters)
    # 404 naming the missing PVC proves valid parameters passed validation.
    catalog_submission(name, parameters, 404, ("PersistentVolumeClaim", missing_job_id))


@pytest.mark.parametrize("value", [True, "1", [], {}], ids=["boolean", "string", "array", "object"])
def test_runtime_parameter_types_rejected_for_the_target_field(
    runtime_algorithm, catalog_submission, value
):
    name, schema, parameters = runtime_algorithm
    parameter = "num_iterations" if name == "lora_grpo" else "num_epochs"
    parameters[parameter] = value
    assert not Draft202012Validator(schema, registry=Registry()).is_valid(parameters)
    catalog_submission(name, parameters, 422, (name, parameter, "must be"))


def test_runtime_nullability_matches_submission(
    runtime_algorithm, catalog_submission, missing_job_id
):
    name, schema, parameters = runtime_algorithm
    parameters["learning_rate"] = None
    allowed = Draft202012Validator(schema, registry=Registry()).is_valid(parameters)
    mentions = ("PersistentVolumeClaim", missing_job_id) if allowed else (name, "learning_rate")
    catalog_submission(name, parameters, 404 if allowed else 422, mentions)


@pytest.mark.parametrize("name", ["sft", "osft", "lora_sft"])
@pytest.mark.parametrize("parameter", ["num_epochs", "effective_batch_size"])
@pytest.mark.parametrize("value", [-1, 0])
def test_nonpositive_counts_rejected_before_storage(catalog_submission, name, parameter, value):
    parameters = {**PARAMETERS[name], parameter: value}
    catalog_submission(name, parameters, 422, (name, parameter, "at least 1"))


@pytest.mark.parametrize("name", ["sft", "osft", "lora_sft"])
def test_nullable_count_defaults_still_reach_storage(api, catalog_submission, missing_job_id, name):
    parameters = {**PARAMETERS[name], "num_epochs": None}
    if name != "osft":
        parameters["effective_batch_size"] = None
    detail = api.object(
        api.request("GET", f"{BASE}/{name}", params={"include": "parameter_schema"})
    )
    Draft202012Validator(detail["parameter_schema"], registry=Registry()).validate(parameters)
    catalog_submission(name, parameters, 404, ("PersistentVolumeClaim", missing_job_id))


@pytest.mark.parametrize(
    "parameter,value", [("backend", "e2e-unsupported-engine"), ("trust_remote_code", True)]
)
def test_runtime_enum_and_constant_constraints(
    runtime_algorithm, catalog_submission, parameter, value
):
    name, schema, parameters = runtime_algorithm
    native = schema["properties"].get(parameter, {})
    if "enum" not in native and "const" not in native:
        pytest.skip(f"{name} runtime has no enum/constant constraint for {parameter}")
    parameters[parameter] = value
    assert not Draft202012Validator(schema, registry=Registry()).is_valid(parameters)
    catalog_submission(name, parameters, 422, (name, parameter, "allowed value(s)"))


def test_each_required_parameter_is_reported(api, catalog_submission):
    checked = 0
    for summary in api.object(api.request("GET", BASE))["algorithms"]:
        if summary["name"] not in PARAMETERS:
            continue
        detail = api.object(
            api.request("GET", summary["details_url"], params={"include": "parameter_schema"})
        )
        schema = detail["parameter_schema"]
        for parameter in schema.get("required", []):
            parameters = deepcopy(PARAMETERS[summary["name"]])
            Draft202012Validator(schema, registry=Registry()).validate(parameters)
            parameters.pop(parameter)
            catalog_submission(
                summary["name"],
                parameters,
                422,
                (summary["name"], "Missing required parameter(s)", parameter),
            )
            checked += 1
    if not checked:
        pytest.skip("Selected runtime has no required parameters for the execution profiles")


def test_managed_and_python_callable_parameters_cannot_be_overridden(
    runtime_algorithm, catalog_submission
):
    name, schema, baseline = runtime_algorithm
    metadata = schema.get("x-training-hub", {})
    unavailable = metadata.get("unavailable_callable_parameters", [])
    for parameter in ["model_path", "nproc_per_node", *unavailable]:
        parameters = {**baseline, parameter: "e2e-cannot-override"}
        assert parameter not in schema["properties"]
        assert not Draft202012Validator(schema, registry=Registry()).is_valid(parameters)
        catalog_submission(name, parameters, 422, (name, parameter, "Unsupported parameter(s)"))


def test_discoverable_algorithm_still_requires_execution_profile(api, catalog_submission):
    names = {a["name"] for a in api.object(api.request("GET", BASE))["algorithms"]}
    if "grpo" not in names:
        pytest.skip("Selected runtime does not advertise full-parameter grpo")
    catalog_submission("grpo", {}, 422, ("grpo", "no execution profile"))


def test_advertised_engine_also_requires_compatible_execution_profile(api, catalog_submission):
    detail = api.object(
        api.request("GET", f"{BASE}/lora_grpo", params={"include": "parameter_schema"})
    )
    parameters = {"backend": "art"}
    schema = detail["parameter_schema"]
    if "art" not in schema["properties"].get("backend", {}).get("enum", []):
        pytest.skip("Selected runtime does not advertise the alternative ART engine")
    Draft202012Validator(schema, registry=Registry()).validate(parameters)
    catalog_submission("lora_grpo", parameters, 422, ("lora_grpo", "requires", "verl"))
