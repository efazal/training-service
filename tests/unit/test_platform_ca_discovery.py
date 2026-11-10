"""Default Helm trust comes from authenticated platform metadata, without CA setup."""

import ssl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from requests.certs import where

from training_service import log_config
from training_service.errors import ApiError
from training_service.jobs import CodeFlareJobBackend
from training_service.main import create_app


@pytest.fixture
def ingress(monkeypatch):
    monkeypatch.setenv("TRAINING_SERVICE_RAY_CA_DISCOVERY", "true")
    for name in ["TRAINING_SERVICE_RAY_CA_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]:
        monkeypatch.delenv(name, raising=False)
    roots = ssl.create_default_context(cafile=where()).get_ca_certs(binary_form=True)
    certificates = [ssl.DER_cert_to_PEM_cert(root) for root in roots[:2]]
    api = Mock()
    api.read_namespaced_config_map.return_value = client.V1ConfigMap(
        data={"ca-bundle.crt": certificates[0]}
    )
    factory = Mock(return_value=api)
    monkeypatch.setattr(log_config.client, "CoreV1Api", factory)

    def load(*, client_configuration):
        client_configuration.api_key["authorization"] = "Bearer service-account-ca-token"

    monkeypatch.setattr(log_config.config, "load_incluster_config", load)
    return api, factory, certificates


def test_default_platform_trust_needs_no_customer_configmap(ingress):
    api, factory, certificates = ingress
    before = client.Configuration.get_default_copy().api_key.copy()
    with TestClient(create_app()) as http:
        assert http.get("/readyz").status_code == 200
        api.read_namespaced_config_map.assert_not_called()
        settings = http.app.state.job_backend.log_configuration()
    assert settings.ca_file != where()
    bundle = Path(settings.ca_file).read_text()
    assert certificates[0] in bundle
    assert Path(where()).read_text().strip() in bundle
    context = ssl.create_default_context(cafile=settings.ca_file)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    api.read_namespaced_config_map.assert_called_once_with(
        "default-ingress-cert", "openshift-config-managed", _request_timeout=(2, 3)
    )
    configuration = factory.call_args.args[0].configuration
    assert configuration.api_key["authorization"] == "Bearer service-account-ca-token"
    assert configuration.verify_ssl is True and configuration.retries == 0
    assert client.Configuration.get_default_copy().api_key == before


def test_ingress_rotation_updates_new_connections_without_unbounded_files(ingress, monkeypatch):
    api, _, certificates = ingress
    now = [100.0]
    monkeypatch.setattr(log_config, "monotonic", lambda: now[0])
    backend = CodeFlareJobBackend()
    path = Path(backend.log_configuration().ca_file)
    original = path.read_text()
    api.read_namespaced_config_map.return_value.data = {"ca-bundle.crt": certificates[1]}
    now[0] = 159.9
    assert backend.log_configuration().ca_file == str(path)
    assert path.read_text() == original
    now[0] = 160.0
    assert backend.log_configuration().ca_file == str(path)
    assert path.read_text() != original and certificates[1] in path.read_text()
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("status", [403, 404, 500])
def test_ca_discovery_failure_is_sanitized_and_retried_without_breaking_readiness(ingress, status):
    api, _, certificates = ingress
    api.read_namespaced_config_map.side_effect = [
        ApiException(status=status, reason="private-upstream-response"),
        client.V1ConfigMap(data={"ca-bundle.crt": certificates[0]}),
    ]
    with TestClient(create_app()) as http:
        with pytest.raises(ApiError) as error:
            http.app.state.job_backend.log_configuration()
        assert error.value.status_code == 503 and error.value.code == "logs_unavailable"
        assert "default-ingress-cert" in error.value.message
        assert "private-upstream-response" not in error.value.message
        assert http.get("/readyz").status_code == 200
        assert Path(http.app.state.job_backend.log_configuration().ca_file).is_file()


@pytest.mark.parametrize("bundle", [None, "", "not PEM", "PRIVATE KEY"])
def test_invalid_platform_certificates_are_rejected_and_can_recover(ingress, bundle):
    api, _, certificates = ingress
    api.read_namespaced_config_map.return_value.data = {"ca-bundle.crt": bundle}
    backend = CodeFlareJobBackend()
    with pytest.raises(ApiError, match="public.*CA bundle"):
        backend.log_configuration()
    api.read_namespaced_config_map.return_value.data = {"ca-bundle.crt": certificates[0]}
    assert Path(backend.log_configuration().ca_file).is_file()


def test_failed_ca_refresh_never_silently_uses_stale_trust(ingress, monkeypatch):
    api, _, _ = ingress
    now = [100.0]
    monkeypatch.setattr(log_config, "monotonic", lambda: now[0])
    backend = CodeFlareJobBackend()
    backend.log_configuration()
    now[0] = 160.0
    api.read_namespaced_config_map.side_effect = ApiException(status=403)
    with pytest.raises(ApiError):
        backend.log_configuration()


@pytest.mark.parametrize(
    "name", ["TRAINING_SERVICE_RAY_CA_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]
)
def test_explicit_trust_bypasses_ingress_discovery(ingress, monkeypatch, name):
    api, _, _ = ingress
    monkeypatch.setenv(name, where())
    assert CodeFlareJobBackend().log_configuration().ca_file == where()
    api.read_namespaced_config_map.assert_not_called()


def test_system_trust_opt_out_bypasses_ingress_discovery(ingress, monkeypatch):
    api, _, _ = ingress
    monkeypatch.setenv("TRAINING_SERVICE_RAY_CA_DISCOVERY", "false")
    assert CodeFlareJobBackend().log_configuration().ca_file == where()
    api.read_namespaced_config_map.assert_not_called()


def test_concurrent_requests_share_one_ca_discovery(ingress):
    api, _, _ = ingress
    backend = CodeFlareJobBackend()
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: backend.log_configuration().ca_file, range(16)))
    assert len(set(paths)) == 1
    assert paths[0] != where()
    api.read_namespaced_config_map.assert_called_once()
