"""Exercise the public discovery routes against a versioned runtime catalog."""

import json
from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, ValidationError

from training_service import catalog
from training_service.auth import AuthenticationUnavailable, CallerIdentity
from training_service.contract import component_validator, json_schema, load_contract
from training_service.errors import ApiError
from training_service.main import create_app
from training_service_api.models.algorithm_details import AlgorithmDetails

HEADERS = {"Authorization": "Bearer test-only-token"}
BASE = "/api/v1/algorithms"


class Authenticator:
    async def authenticate(self, token: str) -> CallerIdentity:
        return CallerIdentity("alice", "uid", ())


@pytest.fixture
def image_catalog() -> dict[str, Any]:
    properties = {
        "learning_rate": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None},
        "num_epochs": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
        "effective_batch_size": {"type": "integer", "default": 8},
        "max_seq_len": {"type": "integer"},
        "warmup_steps": {"type": "integer", "default": 10},
        "trust_remote_code": {"type": "boolean", "const": False, "default": False},
        "target_modules": {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            "default": None,
        },
    }
    algorithms = []
    for name, engine in (
        ("grpo", "verl"),
        ("lora_grpo", "verl"),
        ("lora_sft", "unsloth"),
        ("osft", "mini-trainer"),
        ("sft", "instructlab-training"),
    ):
        props = {"backend": {"type": "string", "enum": [engine], "default": engine}}
        props.update(deepcopy(properties))
        required = []
        if name == "osft":
            props["unfreeze_rank_ratio"] = {"type": "number"}
            props["max_tokens_per_gpu"] = {"type": "integer"}
            required = [
                "unfreeze_rank_ratio",
                "effective_batch_size",
                "max_tokens_per_gpu",
                "max_seq_len",
                "learning_rate",
            ]
        if "grpo" in name:
            props["num_iterations"] = {"type": "integer", "default": 15}
        algorithms.append(
            {
                "name": name,
                "description": "Training Hub Python docstring.\n\nArgs:\n"
                "    warmup_steps: Number of warmup steps before normal training.\n"
                "    target_modules: Modules to which LoRA adapters are applied.\n\n"
                "Examples:\n    result = train(model_path='model')",
                "supported_backends": ["ray"],
                "parameter_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": False,
                    "x-training-hub": {
                        "default_backend": engine,
                        "unavailable_callable_parameters": ["reward_fn"] if "grpo" in name else [],
                        "unavailable_required_parameters": [],
                        "managed_parameters": ["model_path", "data_path", "nnodes"],
                    },
                },
            }
        )
    return {
        "algorithms": algorithms,
        "runtime": {
            "image": "test-runtime@sha256:" + "a" * 64,
            "training_hub_version": "0.8.1",
            "registry_sha256": "b" * 64,
        },
    }


@pytest.fixture
def catalog_file(monkeypatch, image_catalog, tmp_path):
    path = tmp_path / "discovered-catalog.json"
    path.write_text(json.dumps(image_catalog))
    monkeypatch.setenv("TRAINING_SERVICE_CATALOG_PATH", str(path))
    monkeypatch.setenv("TRAINING_RAY_IMAGE", image_catalog["runtime"]["image"])
    return path


@pytest.fixture
def client(catalog_file) -> TestClient:
    return TestClient(create_app(Authenticator()))


def test_list_is_compact_and_preserves_runtime(client: TestClient, image_catalog: dict) -> None:
    response = client.get(BASE, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["runtime"] == image_catalog["runtime"]
    assert len(data["algorithms"]) == 5
    for item in data["algorithms"]:
        assert set(item) == {"name", "description", "supported_backends", "details_url"}
        assert len(item["description"]) <= 240
        assert item["supported_backends"] == ["ray"]
        assert client.get(item["details_url"], headers=HEADERS).status_code == 200
    for text in ("Args:", "Examples:", "parameter_schema", "x-training-hub", "managed_parameters"):
        assert text not in response.text
    assert catalog.validate_catalog(image_catalog) == image_catalog


def test_detail_lists_every_parameter_without_fabricating_defaults(client: TestClient) -> None:
    response = client.get(BASE + "/lora_sft", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    AlgorithmDetails.model_validate(data)
    assert "parameter_schema" not in data
    assert data["required_parameters"] == []
    parameters = data["parameters"]
    assert parameters["learning_rate"] == {
        "type": "number",
        "description": "Learning rate used during training.",
        "nullable": True,
    }
    assert parameters["effective_batch_size"]["default"] == 8
    assert parameters["warmup_steps"]["default"] == 10
    assert parameters["warmup_steps"]["description"].startswith("Number of warmup")
    assert parameters["trust_remote_code"]["default"] is False
    assert parameters["trust_remote_code"]["choices"] == [False]
    assert parameters["target_modules"]["type"] == "array"
    assert parameters["target_modules"]["nullable"] is True
    assert parameters["backend"]["choices"] == ["unsloth"]
    assert data["supported_backends"] == ["ray"]
    assert "distinct from" in parameters["backend"]["description"]
    assert data["parameter_schema_url"] == BASE + "/lora_sft?include=parameter_schema"


@pytest.mark.parametrize("name", ["lora_sft", "sft", "osft", "lora_grpo", "grpo"])
def test_all_details_describe_runtime_without_environment_specific_examples(
    client: TestClient, image_catalog: dict, name: str
) -> None:
    response = client.get(BASE + "/" + name, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    source = next(item for item in image_catalog["algorithms"] if item["name"] == name)
    assert set(data["parameters"]) == set(source["parameter_schema"]["properties"])
    assert data["required_parameters"] == source["parameter_schema"]["required"]
    assert "example_request" not in data
    assert "Qwen/" not in response.text
    assert "pvc://" not in response.text
    assert any("validates algorithms and parameters" in note for note in data["notes"])
    if "grpo" in name:
        assert "reward_fn" not in data["parameters"]
        assert any("callbacks" in note for note in data["notes"])


@pytest.mark.parametrize(
    "name,value,valid",
    [
        ("learning_rate", 0.002, True),
        ("warmup_steps", 0, True),
        ("trust_remote_code", False, True),
        ("learning_rate", "invalid", False),
    ],
)
def test_parameter_examples_come_only_from_runtime_schema(
    client, catalog_file, image_catalog, name, value, valid
):
    schema = image_catalog["algorithms"][2]["parameter_schema"]
    schema["properties"][name]["example"] = value
    catalog_file.write_text(json.dumps(image_catalog))
    response = client.get(BASE + "/lora_sft", headers=HEADERS)
    assert response.status_code == 200
    parameter = response.json()["parameters"][name]
    if valid:
        assert parameter["example"] == value
    else:
        assert "example" not in parameter


def test_raw_schema_is_exact_and_opt_in_only(client: TestClient, image_catalog: dict) -> None:
    plain = client.get(BASE + "/lora_sft", headers=HEADERS).json()
    expanded = client.get(plain["parameter_schema_url"], headers=HEADERS)
    assert expanded.status_code == 200
    data = expanded.json()
    assert data.pop("parameter_schema") == image_catalog["algorithms"][2]["parameter_schema"]
    assert data == plain
    assert catalog.validate_catalog(image_catalog) == image_catalog


def test_discovery_cannot_weaken_raw_submission_validation(image_catalog: dict) -> None:
    assert catalog.validate_catalog(image_catalog) == image_catalog
    schema = image_catalog["algorithms"][3]["parameter_schema"]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate({})
    valid = {
        "unfreeze_rank_ratio": 0.5,
        "effective_batch_size": 8,
        "max_tokens_per_gpu": 1024,
        "max_seq_len": 512,
        "learning_rate": 0.0001,
    }
    Draft202012Validator(schema).validate({**valid, "trust_remote_code": False})
    with pytest.raises(ValidationError) as error:
        Draft202012Validator(schema).validate({**valid, "trust_remote_code": True})
    assert error.value.validator == "const"
    public_summary = {
        "algorithms": [{"name": "sft", "supported_backends": ["ray"], "details_url": BASE + "/sft"}]
    }
    with pytest.raises(ApiError) as error:
        catalog.validate_catalog(public_summary)
    assert error.value.status_code == 503


@pytest.mark.parametrize("suffix", ["", "/lora_sft", "/lora_sft?include=parameter_schema"])
def test_all_discovery_views_require_authentication(client: TestClient, suffix: str) -> None:
    response = client.get(BASE + suffix)
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_authentication_outage_preserves_error_envelope(catalog_file) -> None:
    class Unavailable:
        async def authenticate(self, token: str) -> CallerIdentity:
            raise AuthenticationUnavailable

    client = TestClient(create_app(Unavailable()))
    for path in (BASE, BASE + "/lora_sft"):
        response = client.get(path, headers=HEADERS)
        assert response.status_code == 503
        assert response.json()["code"] == "authentication_unavailable"
        component_validator("ErrorResponse").validate(response.json())


def test_unknown_algorithm_and_invalid_include(client: TestClient) -> None:
    unknown = client.get(BASE + "/does-not-exist", headers=HEADERS)
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "not_found"
    for value in ("all", "", "parameter_schema,other"):
        invalid = client.get(BASE + "/lora_sft", params={"include": value}, headers=HEADERS)
        assert invalid.status_code == 400
        assert invalid.json()["code"] == "bad_request"
        component_validator("ErrorResponse").validate(invalid.json())


@pytest.mark.parametrize(
    "failure", ["stale-image", "missing-runtime", "invalid-json", "invalid-schema", "missing"]
)
def test_catalog_failures_return_503_in_all_views(
    client: TestClient, catalog_file, image_catalog: dict, failure: str
) -> None:
    if failure == "stale-image":
        image_catalog["runtime"]["image"] = "different-image"
        catalog_file.write_text(json.dumps(image_catalog))
    elif failure == "missing-runtime":
        image_catalog.pop("runtime")
        catalog_file.write_text(json.dumps(image_catalog))
    elif failure == "invalid-json":
        catalog_file.write_text("not-json")
    elif failure == "invalid-schema":
        image_catalog["algorithms"][0]["parameter_schema"] = {"type": "unknown-type"}
        catalog_file.write_text(json.dumps(image_catalog))
    else:
        catalog_file.unlink()
    assert client.get("/readyz").status_code == 503
    for suffix in ("", "/lora_sft", "/lora_sft?include=parameter_schema"):
        response = client.get(BASE + suffix, headers=HEADERS)
        assert response.status_code == 503
        assert response.json()["code"] == "backend_unavailable"
        component_validator("ErrorResponse").validate(response.json())


def test_new_algorithms_remain_discoverable_without_hardcoding(
    client: TestClient, catalog_file, image_catalog: dict
) -> None:
    algorithm = image_catalog["algorithms"][0]
    algorithm["name"] = "future_algorithm"
    algorithm["description"] = "A runtime-discovered algorithm. " * 20 + "\n\nArgs:\n..."
    catalog_file.write_text(json.dumps(image_catalog))
    data = client.get(BASE, headers=HEADERS).json()["algorithms"][0]
    assert data["name"] == "future_algorithm"
    assert len(data["description"]) <= 240
    assert "Args:" not in data["description"]
    detail = client.get(data["details_url"], headers=HEADERS)
    assert detail.status_code == 200


@pytest.mark.parametrize("python_only", [False, True])
def test_required_and_python_only_arguments_do_not_produce_invented_requests(
    client: TestClient, catalog_file, image_catalog: dict, python_only: bool
) -> None:
    schema = image_catalog["algorithms"][2]["parameter_schema"]
    if python_only:
        schema["x-training-hub"]["unavailable_required_parameters"] = ["reward_fn"]
    else:
        schema["properties"]["new_required_argument"] = {"type": "string"}
        schema["required"].append("new_required_argument")
    catalog_file.write_text(json.dumps(image_catalog))
    data = client.get(BASE + "/lora_sft", headers=HEADERS).json()
    assert "example_request" not in data
    if python_only:
        assert any("callbacks" in note for note in data["notes"])
    else:
        assert "new_required_argument" in data["required_parameters"]
        assert data["parameters"]["new_required_argument"]["type"] == "string"


def test_swagger_includes_detail_route_and_opt_in(client: TestClient) -> None:
    document = client.get("/openapi.json").json()
    detail = document["paths"]["/algorithms/{name}"]["get"]
    assert detail["operationId"] == "getAlgorithm"
    assert detail["parameters"][1]["schema"]["enum"] == ["parameter_schema"]
    assert set(detail["responses"]) == {"200", "400", "401", "403", "404", "503"}
    assert load_contract()["security"] == [{"OpenShiftBearer": []}]


def test_nullable_conversion_preserves_parameter_descriptor_property_names() -> None:
    properties = {"nullable": {"type": "boolean"}, "type": {"type": "string"}}
    assert json_schema({"type": "object", "properties": properties}) == {
        "type": "object",
        "properties": properties,
    }
    assert json_schema({"type": "integer", "nullable": True}) == {"type": ["integer", "null"]}


def test_catalog_reads_do_not_need_caller_configmap_permissions(client, monkeypatch):
    from training_service import kubernetes_client

    def unexpected(*args, **kwargs):
        pytest.fail("Discovery must not contact Kubernetes to read the catalog")

    monkeypatch.setattr(kubernetes_client.client, "CoreV1Api", unexpected)
    assert client.get(BASE, headers=HEADERS).status_code == 200


def test_empty_catalog_is_successful(client, catalog_file):
    catalog_file.write_text('{"algorithms":[]}')
    response = client.get(BASE, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"algorithms": []}


def test_catalog_validation_uses_explicit_image_over_process_environment(
    monkeypatch, image_catalog
):
    monkeypatch.setenv("TRAINING_RAY_IMAGE", "unrelated-shell-image")
    image = image_catalog["runtime"]["image"]
    assert catalog.validate_catalog(image_catalog, runtime_image=image) == image_catalog
    with pytest.raises(ApiError) as error:
        catalog.validate_catalog(image_catalog, runtime_image="wrong-image")
    assert error.value.status_code == 503


def test_unconfigured_catalog_is_actionable(client, monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINING_SERVICE_CATALOG_PATH", str(tmp_path / "missing.json"))
    response = client.get(BASE, headers=HEADERS)
    assert response.status_code == 503
    assert "discover-algorithms" in response.json()["message"]


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"algorithms": "not-an-array"},
        {
            "algorithms": [
                {"name": "bad/name", "supported_backends": ["ray"], "parameter_schema": {}}
            ]
        },
        {"algorithms": [], "runtime": {"image": "incomplete"}},
    ],
)
def test_malformed_catalog_rejected(invalid):
    with pytest.raises(ApiError) as raised:
        catalog.validate_catalog(invalid)
    assert raised.value.status_code == 503


def test_duplicate_algorithm_names_rejected(image_catalog):
    image_catalog["algorithms"].append(deepcopy(image_catalog["algorithms"][0]))
    with pytest.raises(ApiError):
        catalog.validate_catalog(image_catalog)


def test_discovery_never_fetches_remote_schema_references(client, catalog_file, image_catalog):
    schema = image_catalog["algorithms"][2]["parameter_schema"]
    schema["properties"]["learning_rate"] = {
        "$ref": "https://invalid.example/private-schema",
        "example": 0.001,
    }
    catalog_file.write_text(json.dumps(image_catalog))
    response = client.get(BASE + "/lora_sft?include=parameter_schema", headers=HEADERS)
    assert response.status_code == 200
    assert "example_request" not in response.json()
    assert "example" not in response.json()["parameters"]["learning_rate"]
    assert response.json()["parameter_schema"] == schema


def test_boolean_native_schemas_do_not_break_readable_details(client, catalog_file, image_catalog):
    schema = image_catalog["algorithms"][2]["parameter_schema"]
    schema["properties"]["future_argument"] = True
    schema["properties"]["learning_rate"] = {"anyOf": [False, {"type": "number"}]}
    catalog_file.write_text(json.dumps(image_catalog))
    response = client.get(BASE + "/lora_sft", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["parameters"]["future_argument"]["type"] == "any"


def test_non_ray_algorithms_are_discoverable(client, catalog_file, image_catalog):
    image_catalog["algorithms"][0]["supported_backends"] = ["trainer"]
    catalog_file.write_text(json.dumps(image_catalog))
    response = client.get(BASE, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["algorithms"][0]["supported_backends"] == ["trainer"]


def test_parameter_zero_default_is_not_lost(client, catalog_file, image_catalog):
    image_catalog["algorithms"][2]["parameter_schema"]["properties"]["warmup_steps"]["default"] = 0
    catalog_file.write_text(json.dumps(image_catalog))
    response = client.get(BASE + "/lora_sft", headers=HEADERS)
    assert response.json()["parameters"]["warmup_steps"]["default"] == 0


@pytest.mark.parametrize("name", ["sft", "osft", "lora_sft", "grpo", "lora_grpo"])
def test_submission_uses_the_same_schema_as_discovery(client, image_catalog, name):
    details = client.get(BASE + "/" + name + "?include=parameter_schema", headers=HEADERS).json()
    parameters = {"learning_rate": 0.0001, "warmup_steps": 0}
    if name == "osft":
        parameters.update(
            unfreeze_rank_ratio=0.5,
            effective_batch_size=2,
            max_tokens_per_gpu=128,
            max_seq_len=128,
        )
    Draft202012Validator(details["parameter_schema"]).validate(parameters)
    catalog.validate_parameters(name, parameters, "ray")
    # Defaults stay owned by Training Hub, not filled in by validation.
    assert "backend" not in parameters
    assert "num_epochs" not in parameters


def test_submission_rejects_an_incompatible_execution_backend(client):
    with pytest.raises(ApiError, match="execution backend 'trainer'"):
        catalog.validate_parameters("lora_sft", {}, "trainer")


@pytest.mark.parametrize(
    "definition, value",
    [
        ({"type": "number", "exclusiveMinimum": 0}, 0),
        ({"type": "number", "maximum": 1}, 2),
        ({"type": "string", "enum": ["supported"]}, "unsupported"),
    ],
)
def test_submission_honors_constraints_supplied_by_the_runtime(
    client, catalog_file, image_catalog, definition, value
):
    schema = image_catalog["algorithms"][2]["parameter_schema"]
    schema["properties"]["runtime_option"] = definition
    catalog_file.write_text(json.dumps(image_catalog))
    with pytest.raises(ApiError, match="runtime_option") as raised:
        catalog.validate_parameters("lora_sft", {"runtime_option": value}, "ray")
    assert raised.value.status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nested_nonfinite_parameters_are_rejected(client, catalog_file, image_catalog, value):
    image_catalog["algorithms"][2]["parameter_schema"]["properties"]["options"] = {"type": "object"}
    catalog_file.write_text(json.dumps(image_catalog))
    with pytest.raises(ApiError, match="finite"):
        catalog.validate_parameters("lora_sft", {"options": {"threshold": [value]}}, "ray")


@pytest.mark.parametrize(
    "name,parameters,mentions",
    [
        ("missing_algo", {}, ["missing_algo", "Available algorithms"]),
        ("lora_sft", {"not_in_old_image": 1}, ["lora_sft", "not_in_old_image"]),
        ("lora_sft", {"learning_rate": "private-value"}, ["learning_rate", "lora_sft"]),
        ("lora_sft", {"num_epochs": True}, ["num_epochs", "lora_sft"]),
        ("lora_sft", {"backend": "verl"}, ["backend", "lora_sft"]),
        ("lora_sft", {"trust_remote_code": True}, ["trust_remote_code", "lora_sft"]),
        ("osft", {}, ["Missing required", "unfreeze_rank_ratio", "osft"]),
        ("lora_sft", {"model_path": "/escape"}, ["model_path", "lora_sft"]),
        ("grpo", {"reward_fn": "python"}, ["reward_fn", "grpo"]),
    ],
)
def test_incompatible_submission_is_actionable(client, name, parameters, mentions):
    with pytest.raises(ApiError) as raised:
        catalog.validate_parameters(name, parameters, "ray")
    assert raised.value.status_code == 422
    assert raised.value.code == "validation_error"
    for text in mentions:
        assert text in raised.value.message
    assert "private-value" not in raised.value.message


def test_old_image_can_omit_an_algorithm_and_parameter(client, catalog_file, image_catalog):
    image_catalog["algorithms"] = [a for a in image_catalog["algorithms"] if a["name"] != "grpo"]
    schema = image_catalog["algorithms"][1]["parameter_schema"]
    assert image_catalog["algorithms"][1]["name"] == "lora_sft"
    schema["properties"].pop("warmup_steps")
    catalog_file.write_text(json.dumps(image_catalog))
    names = {a["name"] for a in client.get(BASE, headers=HEADERS).json()["algorithms"]}
    assert "grpo" not in names
    with pytest.raises(ApiError, match="grpo"):
        catalog.validate_parameters("grpo", {}, "ray")
    with pytest.raises(ApiError, match="warmup_steps"):
        catalog.validate_parameters("lora_sft", {"warmup_steps": 0}, "ray")


@pytest.mark.parametrize("python_only", ["unavailable_required_parameters", "api_callable"])
def test_python_only_modes_cannot_be_submitted(client, catalog_file, image_catalog, python_only):
    metadata = image_catalog["algorithms"][2]["parameter_schema"]["x-training-hub"]
    metadata[python_only] = (
        ["rollout_fn"] if python_only == "unavailable_required_parameters" else False
    )
    catalog_file.write_text(json.dumps(image_catalog))
    with pytest.raises(ApiError, match="Python interface"):
        catalog.validate_parameters("lora_sft", {}, "ray")


def test_submission_never_retrieves_remote_schema(client, catalog_file, image_catalog, monkeypatch):

    schema = image_catalog["algorithms"][2]["parameter_schema"]
    schema["properties"]["learning_rate"] = {"$ref": "https://invalid.example/schema"}
    catalog_file.write_text(json.dumps(image_catalog))

    def unexpected(*args, **kwargs):
        pytest.fail("Schema validation must never make an outbound request")

    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    with pytest.raises(ApiError) as raised:
        catalog.validate_parameters("lora_sft", {"learning_rate": 0.001}, "ray")
    assert raised.value.status_code == 503


def test_readiness_and_discovery_share_cached_startup_catalog(catalog_file):
    with TestClient(create_app(Authenticator())) as http:
        assert http.get("/readyz").status_code == 200
        before = http.get(BASE, headers=HEADERS).json()
        catalog_file.unlink()
        assert http.get("/readyz").status_code == 200
        assert http.get(BASE, headers=HEADERS).json() == before
        assert catalog._load_catalog.cache_info().misses == 1


def test_startup_failure_keeps_health_but_not_readiness(catalog_file):
    catalog_file.write_text("invalid-json")
    with TestClient(create_app(Authenticator())) as http:
        assert http.get("/healthz").status_code == 200
        assert http.get("/docs").status_code == 200
        response = http.get("/readyz")
        assert response.status_code == 503
        assert response.json()["code"] == "backend_unavailable"
        assert "discover-algorithms" in response.json()["message"]
        assert http.get(BASE, headers=HEADERS).status_code == 503
