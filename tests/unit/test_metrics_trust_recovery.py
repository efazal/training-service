"""MLflow trust shares validated CA state without blocking warm log requests."""

import ssl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
from requests.certs import where

from training_service import log_config
from training_service.errors import ApiError


@pytest.fixture
def trust(monkeypatch):
    for name in (
        "TRAINING_SERVICE_RAY_CA_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "RAY_ADDRESS",
        "RAY_API_SERVER_ADDRESS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAINING_SERVICE_RAY_CA_DISCOVERY", "true")
    monkeypatch.setenv("TRAINING_SERVICE_RAY_GATEWAY_URL", "https://gateway.example")
    now = [100.0]
    monkeypatch.setattr(log_config, "monotonic", lambda: now[0])
    roots = ssl.create_default_context(cafile=where()).get_ca_certs(binary_form=True)
    certificates = [ssl.DER_cert_to_PEM_cert(root) for root in roots[:2]]
    discover = Mock(return_value=certificates[0])
    monkeypatch.setattr(log_config, "_discover_ingress_ca", discover)
    monkeypatch.setattr(
        log_config,
        "_discover_gateway",
        Mock(side_effect=AssertionError("Trust must not discover the Ray gateway")),
    )
    return log_config.LogBackendResolver(), discover, certificates, now


def test_discovery_failure_reuses_valid_cached_ca(trust):
    resolver, discover, _, now = trust
    path = Path(resolver.trust_file())
    original = path.read_bytes()
    now[0] = 160.0
    discover.side_effect = ApiError(503, "logs_unavailable", "temporary discovery failure")

    assert resolver.trust_file() == str(path)
    assert path.read_bytes() == original
    context = ssl.create_default_context(cafile=str(path))
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_failed_refresh_and_backoff_reject_damaged_cached_ca(trust, damage):
    resolver, discover, _, now = trust
    path = Path(resolver.trust_file())
    if damage == "corrupt":
        path.write_text("damaged local trust")
    else:
        path.unlink()
    now[0] = 160.0
    discover.side_effect = ApiError(503, "logs_unavailable", "temporary discovery failure")

    for _ in range(2):
        with pytest.raises(ApiError, match="CA bundle cannot be loaded"):
            resolver.trust_file()
    assert discover.call_count == 2


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_discovery_retry_is_bounded_and_recovers_at_retry_deadline(trust, warm):
    resolver, discover, certificates, now = trust
    cached = resolver.trust_file() if warm else None
    initial_calls = discover.call_count
    now[0] = 160.0
    discover.side_effect = ApiError(503, "logs_unavailable", "temporary discovery failure")

    def lookup(_):
        if warm:
            assert resolver.trust_file() == cached
        else:
            with pytest.raises(ApiError, match="temporary discovery failure"):
                resolver.trust_file()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lookup, range(16)))
    assert discover.call_count == initial_calls + 1
    now[0] = 164.9
    lookup(None)
    assert discover.call_count == initial_calls + 1

    now[0] = 165.0
    discover.side_effect = None
    discover.return_value = certificates[1]
    recovered = resolver.trust_file()
    assert certificates[1] in Path(recovered).read_text()
    if warm:
        assert recovered == cached
    assert discover.call_count == initial_calls + 2
    now[0] = 224.9
    assert resolver.trust_file() == recovered
    assert discover.call_count == initial_calls + 2
    now[0] = 225.0
    assert resolver.trust_file() == recovered
    assert discover.call_count == initial_calls + 3


@pytest.mark.parametrize("source", ["trust_file", "configuration"])
def test_warm_configuration_and_trust_do_not_wait_for_ca_discovery(trust, source):
    resolver, discover, certificates, now = trust
    original = resolver.configuration()
    now[0] = 160.0
    entered, release = Event(), Event()

    def slow_discovery():
        entered.set()
        assert release.wait(5)
        return certificates[1]

    discover.side_effect = slow_discovery
    with ThreadPoolExecutor(max_workers=2) as pool:
        refreshing = pool.submit(getattr(resolver, source))
        try:
            assert entered.wait(2)
            warm = pool.submit(lambda: (resolver.configuration(), resolver.trust_file()))
            assert warm.result(timeout=2) == (original, original.ca_file)
            assert not refreshing.done()
        finally:
            release.set()
        refreshing.result(timeout=2)
    assert certificates[1] in Path(original.ca_file).read_text()
    assert discover.call_count == 2


@pytest.mark.parametrize(
    "name", ["TRAINING_SERVICE_RAY_CA_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]
)
def test_explicit_ca_is_revalidated_without_cached_platform_fallback(
    trust, monkeypatch, tmp_path, name
):
    resolver, discover, _, now = trust
    resolver.trust_file()
    now[0] = 160.0
    discover.side_effect = ApiError(503, "logs_unavailable", "temporary discovery failure")
    explicit = tmp_path / "explicit.crt"
    explicit.write_bytes(Path(where()).read_bytes())
    monkeypatch.setenv(name, str(explicit))

    assert resolver.trust_file() == str(explicit)
    explicit.write_text("damaged explicit trust")
    with pytest.raises(ApiError, match="CA bundle cannot be loaded"):
        resolver.trust_file()
    discover.assert_called_once()
