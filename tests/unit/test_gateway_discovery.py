"""Resolve published platform metadata independently of caller workload credentials."""

import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from requests.certs import where
from urllib3.exceptions import HTTPError

from training_service import log_config
from training_service.jobs import CodeFlareJobBackend
from training_service.main import create_app


def ready_gateway():
    return {
        "metadata": {"generation": 2},
        "status": {
            "observedGeneration": 2,
            "phase": "Ready",
            "domain": "rh-ai.apps.example.com",
            "conditions": [
                {"type": "Ready", "status": "True"},
                {"type": "GatewayConfigReady", "status": "True"},
            ],
        },
    }


@pytest.fixture
def platform(monkeypatch):
    monkeypatch.delenv("TRAINING_SERVICE_RAY_GATEWAY_URL")
    api = Mock()
    api.get_cluster_custom_object.return_value = ready_gateway()
    factory = Mock(return_value=api)
    monkeypatch.setattr(log_config.client, "CustomObjectsApi", factory)

    def load(*, client_configuration):
        client_configuration.host = "https://kubernetes.example.test"
        client_configuration.api_key["authorization"] = "Bearer service-account-test-token"

    loader = Mock(side_effect=load)
    monkeypatch.setattr(log_config.config, "load_incluster_config", loader)
    return api, factory, loader


def test_startup_discovers_once_with_isolated_service_identity(platform, caplog):
    api, factory, loader = platform
    before = client.Configuration.get_default_copy().api_key.copy()
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    app = create_app()
    with TestClient(app) as http:
        assert http.get("/readyz").status_code == 200
        assert http.get("/readyz").status_code == 200
        assert (
            app.state.job_backend.log_configuration().gateway_url
            == "https://rh-ai.apps.example.com"
        )
    api.get_cluster_custom_object.assert_called_once_with(
        "services.platform.opendatahub.io",
        "v1alpha1",
        "gatewayconfigs",
        "default-gateway",
        _request_timeout=(2, 3),
    )
    loader.assert_called_once()
    configuration = factory.call_args.args[0].configuration
    assert configuration.api_key["authorization"] == "Bearer service-account-test-token"
    assert configuration.verify_ssl is True and configuration.retries == 0
    assert client.Configuration.get_default_copy().api_key == before
    assert "GatewayConfig/default-gateway status.domain" in caplog.text
    assert "https://rh-ai.apps.example.com" in caplog.text
    assert "service-account-test-token" not in caplog.text


@pytest.mark.parametrize("override", ["https://custom.example", "https://custom.example/"])
def test_explicit_override_bypasses_platform_discovery(platform, monkeypatch, override):
    api, _, loader = platform
    monkeypatch.setenv("TRAINING_SERVICE_RAY_GATEWAY_URL", override)
    assert CodeFlareJobBackend().log_configuration().gateway_url == "https://custom.example"
    loader.assert_not_called()
    api.get_cluster_custom_object.assert_not_called()


def test_invalid_explicit_override_is_not_replaced_by_discovery(platform, monkeypatch):
    api, _, loader = platform
    monkeypatch.setenv("TRAINING_SERVICE_RAY_GATEWAY_URL", "https://user:secret@custom.example")
    http = TestClient(create_app())
    response = http.get("/readyz")
    assert response.status_code == 503
    assert "user:secret" not in response.text
    loader.assert_not_called()
    api.get_cluster_custom_object.assert_not_called()


def test_not_ready_retries_then_recovers_without_restart(platform):
    api, *_ = platform
    waiting = ready_gateway()
    waiting["status"]["phase"] = "Progressing"
    api.get_cluster_custom_object.side_effect = [waiting, waiting, ready_gateway()]
    with TestClient(create_app()) as http:
        assert http.get("/healthz").status_code == 200
        first = http.get("/readyz")
        assert first.status_code == 503 and "not ready" in first.json()["message"]
        assert http.get("/readyz").status_code == 200
        assert http.get("/readyz").status_code == 200
    assert api.get_cluster_custom_object.call_count == 3


@pytest.mark.parametrize("status", [403, 404, 500])
def test_discovery_api_errors_are_actionable_sanitized_and_retryable(platform, status):
    api, *_ = platform
    api.get_cluster_custom_object.side_effect = [
        ApiException(status=status, reason="sensitive-backend-response"),
        ready_gateway(),
    ]
    http = TestClient(create_app())
    response = http.get("/readyz")
    assert response.status_code == 503
    assert response.json()["code"] == "logs_unavailable"
    assert "TRAINING_SERVICE_RAY_GATEWAY_URL" in response.json()["message"]
    assert "sensitive-backend-response" not in response.text
    assert http.get("/healthz").status_code == 200
    assert http.get("/readyz").status_code == 200


@pytest.mark.parametrize(
    "error",
    [
        ConfigException("sensitive-kubeconfig"),
        HTTPError("sensitive-network"),
        OSError("sensitive-path"),
    ],
)
def test_no_kubeconfig_or_unverified_fallback_when_service_discovery_fails(platform, error):
    api, _, loader = platform
    loader.side_effect = error
    response = TestClient(create_app()).get("/readyz")
    assert response.status_code == 503
    assert "in-cluster service account" in response.json()["message"]
    assert "sensitive" not in response.text
    api.get_cluster_custom_object.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"observedGeneration": 1},
        {"observedGeneration": None},
        {"phase": "Progressing"},
        {"conditions": []},
        {"conditions": [{"type": "Ready", "status": "False"}]},
        {
            "conditions": [
                {"type": "Ready", "status": "True"},
                {"type": "GatewayConfigReady", "status": "False"},
            ]
        },
    ],
)
def test_stale_or_unready_metadata_is_never_used(platform, change):
    api, *_ = platform
    api.get_cluster_custom_object.return_value["status"].update(change)
    response = TestClient(create_app()).get("/readyz")
    assert response.status_code == 503
    assert "not ready" in response.json()["message"]


@pytest.mark.parametrize(
    "domain",
    [
        None,
        "",
        "https://gateway.example",
        "user:secret@gateway.example",
        "gateway.example/path",
        "gateway.example:6443",
        "gateway.example?q=1",
        "a..example",
        "-a.example",
        "a-.example",
        "a" * 64 + ".example",
        "a." * 127 + "example",
    ],
)
def test_invalid_published_domain_is_rejected_without_guessing(platform, domain):
    api, *_ = platform
    api.get_cluster_custom_object.return_value["status"]["domain"] = domain
    response = TestClient(create_app()).get("/readyz")
    assert response.status_code == 503
    assert "status.domain" in response.json()["message"]
    assert "user:secret" not in response.text


def test_resolved_endpoint_stays_pinned_until_backend_restart(platform):
    api, *_ = platform
    backend = CodeFlareJobBackend()
    original = backend.log_configuration()
    changed = deepcopy(ready_gateway())
    changed["status"]["domain"] = "new.apps.example.com"
    api.get_cluster_custom_object.return_value = changed
    assert backend.log_configuration().gateway_url == original.gateway_url
    api.get_cluster_custom_object.assert_called_once()
    assert CodeFlareJobBackend().log_configuration().gateway_url == "https://new.apps.example.com"


def test_concurrent_first_requests_share_one_discovery(platform):
    api, *_ = platform
    backend = CodeFlareJobBackend()
    with ThreadPoolExecutor(max_workers=8) as pool:
        endpoints = list(pool.map(lambda _: backend.log_configuration().gateway_url, range(16)))
    assert set(endpoints) == {"https://rh-ai.apps.example.com"}
    api.get_cluster_custom_object.assert_called_once()


def test_discovered_gateway_keeps_loading_rotated_ca_without_rediscovery(
    platform, monkeypatch, tmp_path
):
    api, *_ = platform
    path = tmp_path / "ca.crt"
    path.write_bytes(Path(where()).read_bytes())
    monkeypatch.setenv("TRAINING_SERVICE_RAY_CA_FILE", str(path))
    http = TestClient(create_app())
    assert http.get("/readyz").status_code == 200
    path.write_text("invalid replacement")
    assert http.get("/readyz").status_code == 503
    path.write_bytes(Path(where()).read_bytes())
    assert http.get("/readyz").status_code == 200
    api.get_cluster_custom_object.assert_called_once()
