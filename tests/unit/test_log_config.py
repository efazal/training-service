"""Permanent log configuration fails readiness before an active job is needed."""

import ssl
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from requests.certs import where

from training_service.errors import ApiError
from training_service.log_config import log_backend_config
from training_service.main import create_app


@pytest.mark.parametrize(
    "gateway",
    [
        "http://gateway.example",
        "https://user:pass@gateway.example",
        "https://gateway.example/path",
        "https://gateway.example?q=1",
        "https://gateway.example#fragment",
        "https://gateway.example:0",
        "https://gateway.example:99999",
        "https://gateway.example/ /",
        "https://gateway.example:bad",
    ],
)
def test_invalid_gateway_fails_readiness_but_not_liveness(monkeypatch, gateway):
    monkeypatch.setenv("TRAINING_SERVICE_RAY_GATEWAY_URL", gateway)
    with TestClient(create_app()) as http:
        assert http.get("/healthz").status_code == 200
        response = http.get("/readyz")
        assert response.status_code == 503
        assert response.json()["code"] == "logs_unavailable"
        assert "TRAINING_SERVICE_RAY_GATEWAY_URL" in response.json()["message"]
        assert "user:pass" not in response.text


def test_custom_ca_rotation_is_read_without_restart(monkeypatch, tmp_path):
    path = tmp_path / "ca.crt"
    path.write_bytes(Path(where()).read_bytes())
    monkeypatch.setenv("TRAINING_SERVICE_RAY_CA_FILE", str(path))
    http = TestClient(create_app())
    assert http.get("/readyz").status_code == 200
    assert log_backend_config().ca_file == str(path)
    path.write_text("invalid replacement")
    assert http.get("/readyz").status_code == 503
    path.write_bytes(Path(where()).read_bytes())
    assert http.get("/readyz").status_code == 200


@pytest.mark.parametrize("setting", ["RAY_ADDRESS", "RAY_API_SERVER_ADDRESS"])
def test_conflicting_sdk_address_fails_configuration(monkeypatch, setting):
    monkeypatch.setenv(setting, "https://other.example")
    with pytest.raises(ApiError, match="Unset"):
        log_backend_config()


def test_default_trust_is_explicit_and_verifies_certificates():
    settings = log_backend_config()
    assert settings.ca_file == where()
    context = ssl.create_default_context(cafile=settings.ca_file)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
