"""Non-submitting checks over the deployed HTTP API, not FastAPI TestClient."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from tests.e2e.algorithms import ALGORITHMS


@pytest.mark.parametrize("path", ["/healthz", "/readyz"])
def test_probes(api, path):
    response = api.request("GET", path, authenticated=False)
    assert api.object(response) == {"status": "ok"}


def test_swagger_and_openapi(api):
    swagger = api.request("GET", "/docs", authenticated=False)
    assert "text/html" in swagger.headers.get("content-type", "")
    assert "SwaggerUIBundle" in swagger.text
    document = api.object(api.request("GET", "/openapi.json", authenticated=False))
    assert document["openapi"].startswith("3.")
    paths = document["paths"]
    assert {"get", "post"} <= paths["/projects/{project}/training-jobs"].keys()
    assert {"get", "delete"} <= paths["/projects/{project}/training-jobs/{job_id}"].keys()
    assert "get" in paths["/projects/{project}/training-jobs/{job_id}/logs"]
    canonical = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "docs/api/openapi.yaml").read_text()
    )
    assert document["components"]["schemas"] == canonical["components"]["schemas"]
    if directory := os.getenv("TRAINING_SERVICE_EVIDENCE_DIR"):
        (Path(directory) / "contract.json").write_text(
            json.dumps(
                {
                    "openapi": document["openapi"],
                    "api_version": document["info"]["version"],
                    "sha256": hashlib.sha256(
                        json.dumps(document, sort_keys=True).encode()
                    ).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )


@pytest.mark.parametrize(
    "method,suffix",
    [
        ("GET", ""),
        ("POST", ""),
        ("GET", "/missing"),
        ("DELETE", "/missing"),
        ("GET", "/missing/logs"),
    ],
)
def test_bearer_token_required(api, missing_job_id, method, suffix):
    suffix = suffix.replace("missing", missing_job_id)
    response = api.request(method, api.jobs + suffix, expected=401, authenticated=False, json={})
    api.error(response, "unauthorized")
    assert response.headers.get("www-authenticate", "").lower() == "bearer"


@pytest.mark.parametrize(
    "method,suffix",
    [
        ("GET", ""),
        ("POST", ""),
        ("GET", "/missing"),
        ("DELETE", "/missing"),
        ("GET", "/missing/logs"),
    ],
)
def test_invalid_token_rejected(api, missing_job_id, method, suffix):
    suffix = suffix.replace("missing", missing_job_id)
    response = api.request(
        method,
        api.jobs + suffix,
        expected=401,
        headers={"Authorization": "Bearer e2e-invalid-token"},
        json={},
    )
    api.error(response, "unauthorized")


@pytest.mark.parametrize("authorization", ["Bearer", "Basic e2e-invalid", "Bearer a b"])
def test_malformed_authorization_rejected(api, authorization):
    response = api.request("GET", api.jobs, expected=401, headers={"Authorization": authorization})
    api.error(response, "unauthorized")


def test_list_jobs(api):
    api.page({"limit": 5})


@pytest.mark.parametrize("params", [{"status": "NOT_A_STATUS"}, {"limit": 0}])
def test_invalid_list_query(api, params):
    api.error(api.request("GET", api.jobs, expected=400, params=params), "bad_request")


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_missing_job(api, missing_job_id, method):
    path = f"{api.jobs}/{missing_job_id}"
    api.error(
        api.request(method, path, expected=404),
        "not_found",
        mentions=("Training job", f"'{missing_job_id}'", f"project '{api.project}'", "not found"),
    )


def test_missing_project_has_context(api, missing_job_id):
    response = api.request(
        "GET", f"/api/v1/projects/{missing_job_id}/training-jobs", expected=(403, 404)
    )
    if response.status_code == 403:
        api.error(response, "forbidden")
        pytest.skip("Caller cannot authorize listing in the nonexistent project; 404 not exercised")
    api.error(response, "not_found", mentions=(f"Project '{missing_job_id}'", "not found"))


@pytest.mark.parametrize(
    "scheme,kind", [("pvc", "PersistentVolumeClaim"), ("hf", "Secret"), ("s3", "Secret")]
)
def test_missing_submission_resource_has_context(api, missing_job_id, pytestconfig, scheme, kind):
    # All references are random and nonexistent. Lookup must fail before job creation.
    paths = {"pvc": "model/", "hf": "example/model", "s3": "bucket/model/"}
    payload = {
        "name": "e2e-" + missing_job_id[-16:],
        "model": {"uri": f"{scheme}://{missing_job_id}/{paths[scheme]}"},
        "dataset": {"uri": f"pvc://{missing_job_id}/dataset.jsonl"},
        "algorithm": {"name": "lora_sft"},
    }
    try:
        response = api.request("POST", api.jobs, expected=404, json=payload)
        api.error(
            response,
            "not_found",
            mentions=(kind, f"'{missing_job_id}'", f"project '{api.project}'", "not found"),
        )
        api.error(api.request("GET", f"{api.jobs}/{payload['name']}", expected=404), "not_found")
    finally:
        # A regression or lost POST response must not silently leave a created job behind.
        api.cleanup(payload["name"], pytestconfig.getoption("cleanup_timeout"), payload)


@pytest.mark.parametrize("follow", [False, True])
def test_missing_job_logs_are_json_not_a_stream(api, missing_job_id, follow):
    response = api.request(
        "GET", f"{api.jobs}/{missing_job_id}/logs", expected=404, params={"follow": follow}
    )
    api.error(
        response,
        "not_found",
        mentions=("Training job", f"'{missing_job_id}'", f"project '{api.project}'", "not found"),
    )
    assert response.headers.get("content-type", "").startswith("application/json")


@pytest.mark.parametrize("follow", [False, True])
def test_missing_ray_submission_is_not_retryable(api, pytestconfig, follow):
    job_id = pytestconfig.getoption("missing_submission_job")
    if not job_id:
        pytest.skip("Pass --missing-submission-job with a prepared terminal RayJob")
    path = f"{api.jobs}/{job_id}"
    detail = api.object(api.request("GET", path))
    assert detail["status"] in {"SUCCEEDED", "FAILED", "STOPPED"}, (
        "The missing-submission fixture must be terminal"
    )
    response = api.request("GET", path + "/logs", expected=404, params={"follow": follow})
    api.error(
        response,
        "not_found",
        mentions=("submission", f"job '{job_id}'", f"project '{api.project}'", "not found"),
    )
    assert response.headers.get("content-type", "").startswith("application/json")
    assert "retry later" not in api.object(response)["message"].lower()


@pytest.mark.parametrize("suffix", ["", "/missing", "/missing/logs"])
def test_invalid_kubernetes_url_fails_closed(api, pytestconfig, missing_job_id, suffix):
    if not pytestconfig.getoption("expect_invalid_kubernetes_url"):
        pytest.skip("Requires an isolated deployment configured with an invalid Kubernetes URL")
    suffix = suffix.replace("missing", missing_job_id)
    response = api.request("GET", api.jobs + suffix, expected=503)
    api.error(response, "backend_unavailable")
    assert "HTTPS" in api.object(response)["message"]


@pytest.mark.parametrize(
    "body,code,status", [(b"{}", "validation_error", 422), (b"{", "bad_request", 400)]
)
def test_invalid_submission_body(api, body, code, status):
    response = api.request(
        "POST",
        api.jobs,
        expected=status,
        content=body,
        headers={"Content-Type": "application/json"},
    )
    api.error(response, code)


def test_project_isolation(api, forbidden_project):
    if forbidden_project == api.project:
        raise pytest.UsageError("--forbidden-project must be a different namespace name")
    response = api.request(
        "GET", f"/api/v1/projects/{forbidden_project}/training-jobs", expected=403
    )
    api.error(response, "forbidden")


def test_unsupported_content_type(api):
    api.error(
        api.request(
            "POST", api.jobs, expected=415, content=b"{}", headers={"Content-Type": "text/plain"}
        ),
        "unsupported_media_type",
    )


@pytest.mark.parametrize(
    "params",
    [
        {"limit": -1},
        {"page_token": "not-a-valid-token"},
        {"status": "running"},
    ],
)
def test_more_invalid_list_queries(api, params):
    code = "invalid_page_token" if "page_token" in params else "bad_request"
    api.error(api.request("GET", api.jobs, expected=400, params=params), code)


@pytest.mark.parametrize("follow,tail", [("invalid", "10"), ("false", "0"), ("true", "-1")])
def test_invalid_log_queries(api, missing_job_id, follow, tail):
    api.error(
        api.request(
            "GET",
            f"{api.jobs}/{missing_job_id}/logs",
            expected=400,
            params={"follow": follow, "tail_lines": tail},
        ),
        "bad_request",
    )


@pytest.mark.parametrize(
    "variation",
    [
        "unknown_algorithm",
        "unsupported_parameter",
        "bool_learning_rate",
        "string_epochs",
        "negative_epochs",
        "zero_batch",
        "invalid_cpu",
        "invalid_memory",
        "zero_nodes",
        "traversal",
        "unknown_field",
        "invalid_backend",
        "kft",
        "empty_queue",
        "hf_output",
    ],
)
def test_rejected_submission_creates_no_job(api, missing_job_id, pytestconfig, variation):
    payload = {
        "name": "e2e-" + missing_job_id[-16:],
        "model": {"uri": "pvc://e2e-missing-input/model/"},
        "dataset": {"uri": "pvc://e2e-missing-input/data.jsonl"},
        "algorithm": {"name": "lora_sft", "parameters": {}},
    }
    changes = {
        "unknown_algorithm": {"name": "not-an-algorithm"},
        # Accepted parameters belong to the runtime catalog, not a hardcoded API list.
        "unsupported_parameter": {"parameters": {"unsupported_e2e_parameter": 128}},
        "bool_learning_rate": {"parameters": {"learning_rate": True}},
        "string_epochs": {"parameters": {"num_epochs": "1"}},
        "negative_epochs": {"parameters": {"num_epochs": -1}},
        "zero_batch": {"parameters": {"effective_batch_size": 0}},
    }
    payload["algorithm"].update(changes.get(variation, {}))
    resources = {
        "invalid_cpu": {"cpus_per_node": "four"},
        "invalid_memory": {"memory_per_node": "lots"},
        "zero_nodes": {"num_nodes": 0},
    }
    if variation in resources:
        payload["resources"] = resources[variation]
    if variation == "traversal":
        payload["model"]["uri"] = "pvc://e2e-missing-input/%2e%2e/model"
    if variation == "unknown_field":
        payload["not_in_contract"] = True
    if variation in {"kft", "invalid_backend"}:
        payload["backend"] = "kft" if variation == "kft" else "unknown"
    if variation == "empty_queue":
        payload["queue_name"] = ""
    if variation == "hf_output":
        payload["output"] = {"destination_uri": "hf://e2e-missing-input/org/repository"}
    try:
        response = api.request(
            "POST", api.jobs, expected=501 if variation == "kft" else 422, json=payload
        )
        api.error(response, "not_implemented" if variation == "kft" else "validation_error")
        api.error(api.request("GET", f"{api.jobs}/{payload['name']}", expected=404), "not_found")
    finally:
        api.cleanup(payload["name"], pytestconfig.getoption("cleanup_timeout"), payload)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("invalid", ["unsupported", "malformed"])
def test_algorithm_parameters_rejected_before_submission(
    api, missing_job_id, pytestconfig, algorithm, invalid
):
    """The real API must reject bad per-algorithm overrides without creating a RayJob."""
    bad_values = {
        "sft": {"num_epochs": "one"},
        "osft": {"unfreeze_rank_ratio": "half"},
        "lora_sft": {"logging_steps": "one"},
        "lora_grpo": {"group_size": "four"},
    }
    parameters = (
        {"not_a_training_hub_parameter": 1} if invalid == "unsupported" else bad_values[algorithm]
    )
    payload = {
        "name": "e2e-" + missing_job_id[-16:],
        "model": {"uri": "pvc://e2e-missing-input/model/"},
        "dataset": {"uri": "pvc://e2e-missing-input/data.jsonl"},
        "algorithm": {"name": algorithm, "parameters": parameters},
    }
    try:
        response = api.request("POST", api.jobs, expected=422, json=payload)
        api.error(response, "validation_error")
        assert algorithm in api.object(response)["message"]
        api.error(api.request("GET", f"{api.jobs}/{payload['name']}", expected=404), "not_found")
    finally:
        api.cleanup(payload["name"], pytestconfig.getoption("cleanup_timeout"), payload)
