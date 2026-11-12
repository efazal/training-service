"""Resolve and validate log backend settings for each new log connection."""

import logging
import os
import re
import ssl
import threading
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from urllib.parse import urlsplit

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from requests.certs import where
from urllib3.exceptions import HTTPError

from training_service.errors import ApiError

# Use the ASGI server logger so resolution is visible with the default container logging.
_LOG = logging.getLogger("uvicorn.error")
_GATEWAY_GROUP = "services.platform.opendatahub.io"
_GATEWAY_NAME = "default-gateway"
_DISCOVERY_TIMEOUT = (2, 3)
_GATEWAY_CACHE_SECONDS = 60
_DISCOVERY_RETRY_SECONDS = 5
_INGRESS_CA_NAMESPACE = "openshift-config-managed"
_INGRESS_CA_NAME = "default-ingress-cert"
_DOMAIN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
)


def _discover_gateway() -> str:
    """Read published platform metadata with an isolated service-account client."""
    try:
        configuration = client.Configuration()
        config.load_incluster_config(client_configuration=configuration)
        configuration.verify_ssl = True
        configuration.retries = 0
        with client.ApiClient(configuration) as api:
            gateway = client.CustomObjectsApi(api).get_cluster_custom_object(
                _GATEWAY_GROUP,
                "v1alpha1",
                "gatewayconfigs",
                _GATEWAY_NAME,
                _request_timeout=_DISCOVERY_TIMEOUT,
            )
    except (ApiException, ConfigException, HTTPError, OSError) as error:
        if isinstance(error, ApiException) and error.status == 403:
            message = (
                "Gateway discovery is forbidden. Grant the service account get access to "
                "GatewayConfig/default-gateway, or set TRAINING_SERVICE_RAY_GATEWAY_URL."
            )
        elif isinstance(error, ApiException) and error.status == 404:
            message = (
                "GatewayConfig/default-gateway was not found. Wait for platform setup, "
                "or set TRAINING_SERVICE_RAY_GATEWAY_URL for a custom platform."
            )
        else:
            message = (
                "Cannot discover the platform gateway using the in-cluster service account. "
                "Check Kubernetes connectivity and gateway discovery RBAC, or set "
                "TRAINING_SERVICE_RAY_GATEWAY_URL."
            )
        raise ApiError(503, "logs_unavailable", message) from None
    metadata = gateway.get("metadata", {})
    status = gateway.get("status", {})
    ready = [
        condition
        for condition in status.get("conditions", [])
        if condition.get("type") in {"Ready", "GatewayConfigReady"}
    ]
    generation = metadata.get("generation")
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or status.get("observedGeneration") != generation
        or status.get("phase") != "Ready"
        or not ready
        or any(condition.get("status") != "True" for condition in ready)
    ):
        raise ApiError(
            503,
            "logs_unavailable",
            "GatewayConfig/default-gateway is not ready for its current generation; "
            "gateway discovery will retry on the next log request.",
        )
    domain = status.get("domain")
    if not isinstance(domain, str) or len(domain) > 253 or _DOMAIN.fullmatch(domain) is None:
        raise ApiError(
            503,
            "logs_unavailable",
            "GatewayConfig/default-gateway has no valid status.domain. "
            "Wait for the platform gateway configuration to become ready.",
        )
    return "https://" + domain


def _discover_ingress_ca() -> str:
    """Read only the platform's published public ingress certificates, never Secrets."""
    try:
        configuration = client.Configuration()
        config.load_incluster_config(client_configuration=configuration)
        configuration.verify_ssl = True
        configuration.retries = 0
        with client.ApiClient(configuration) as api:
            configmap = client.CoreV1Api(api).read_namespaced_config_map(
                _INGRESS_CA_NAME, _INGRESS_CA_NAMESPACE, _request_timeout=_DISCOVERY_TIMEOUT
            )
    except (ApiException, ConfigException, HTTPError, OSError):
        raise ApiError(
            503,
            "logs_unavailable",
            "Cannot read the platform public ingress CA from "
            "openshift-config-managed/default-ingress-cert. Check platform setup and the "
            "chart's log-trust RoleBinding, or configure logBackend.trustedCA for a custom issuer.",
        ) from None
    bundle = (configmap.data or {}).get("ca-bundle.crt")
    try:
        if not isinstance(bundle, str) or not bundle or "PRIVATE KEY" in bundle:
            raise ValueError
        ssl.create_default_context(cadata=bundle)
    except (OSError, ValueError):
        raise ApiError(
            503,
            "logs_unavailable",
            "The platform public ingress CA bundle in "
            "openshift-config-managed/default-ingress-cert is missing or invalid.",
        ) from None
    return bundle


class LogBackendResolver:
    """Refresh platform metadata without discarding the last validated configuration."""

    def __init__(self) -> None:
        self._gateway: str | None = None
        self._refresh_at = 0.0
        self._ca_directory: TemporaryDirectory[str] | None = None
        self._ca_file: str | None = None
        self._ca_refresh_at = 0.0
        self._settings: LogBackendConfig | None = None
        self._failure: ApiError | None = None
        self._retry_at = 0.0
        self._refreshing = False
        self._condition = threading.Condition()

    def _platform_ca_file(self) -> str | None:
        if os.getenv("TRAINING_SERVICE_RAY_CA_DISCOVERY") != "true" or any(
            os.getenv(name)
            for name in ("TRAINING_SERVICE_RAY_CA_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
        ):
            return None
        if self._ca_file is None or monotonic() >= self._ca_refresh_at:
            bundle = _discover_ingress_ca()
            try:
                if self._ca_directory is None:
                    self._ca_directory = TemporaryDirectory(prefix="training-service-log-ca-")
                directory = Path(self._ca_directory.name)
                pending = directory / "ca.next"
                # Keep ordinary public roots as well as the platform's private ingress issuer.
                pending.write_text(Path(where()).read_text().rstrip() + "\n" + bundle)
                ssl.create_default_context(cafile=str(pending))
                target = directory / "ca.crt"
                pending.replace(target)
            except (OSError, ValueError):
                raise ApiError(
                    503,
                    "logs_unavailable",
                    "Cannot prepare the platform public CA bundle. Ensure the service's "
                    "temporary directory is writable (writableTemp.enabled in the chart).",
                ) from None
            self._ca_file = str(target)
            self._ca_refresh_at = monotonic() + _GATEWAY_CACHE_SECONDS
        return self._ca_file

    def configuration(self) -> "LogBackendConfig":
        with self._condition:
            # Only cold-start callers wait. Warm callers can use the validated cache
            # while one request refreshes it, without holding a lock over network I/O.
            while self._refreshing and self._settings is None:
                self._condition.wait()
            cached = self._settings
            now = monotonic()
            due = (
                cached is None
                or now >= self._refresh_at
                or (self._ca_file is not None and now >= self._ca_refresh_at)
            )
            refresh = due and not self._refreshing and now >= self._retry_at
            if refresh:
                self._refreshing = True
            elif cached is None:
                assert self._failure is not None
                raise ApiError(503, self._failure.code, self._failure.message)
        if not refresh:
            assert cached is not None
            return log_backend_config(cached.gateway_url, ca_file=cached.ca_file)

        try:
            override = os.getenv("TRAINING_SERVICE_RAY_GATEWAY_URL")
            gateway = override or (
                self._gateway
                if self._gateway is not None and monotonic() < self._refresh_at
                else _discover_gateway()
            )
            resolved = log_backend_config(gateway, ca_file=self._platform_ca_file())
        except ApiError as error:
            with self._condition:
                self._failure = error
                self._retry_at = monotonic() + _DISCOVERY_RETRY_SECONDS
            if cached is None:
                raise
            _LOG.warning(
                "Log configuration refresh failed; using cached gateway and CA, "
                "retrying in %s seconds: %s",
                _DISCOVERY_RETRY_SECONDS,
                error.message,
            )
            # Revalidate local trust and explicit overrides; cached metadata never
            # disables TLS verification or hides a damaged/deleted trust file.
            return log_backend_config(cached.gateway_url, ca_file=cached.ca_file)
        else:
            with self._condition:
                changed = self._gateway != resolved.gateway_url
                self._gateway = resolved.gateway_url
                self._settings = resolved
                self._failure = None
                self._retry_at = 0.0
                self._refresh_at = (
                    float("inf") if override else monotonic() + _GATEWAY_CACHE_SECONDS
                )
            source = (
                "TRAINING_SERVICE_RAY_GATEWAY_URL"
                if override
                else "GatewayConfig/default-gateway status.domain"
            )
            if changed:
                _LOG.info("Resolved log gateway %s from %s", resolved.gateway_url, source)
            return resolved
        finally:
            with self._condition:
                self._refreshing = False
                self._condition.notify_all()


@dataclass(frozen=True)
class LogBackendConfig:
    gateway_url: str
    ca_file: str


def log_backend_config(
    gateway_url: str | None = None, *, ca_file: str | None = None
) -> LogBackendConfig:
    """Resolve trust on each new connection so mounted CA updates take effect."""
    gateway = (
        gateway_url
        if gateway_url is not None
        else os.getenv("TRAINING_SERVICE_RAY_GATEWAY_URL", "")
    ).rstrip("/")
    try:
        parsed = urlsplit(gateway)
        valid = (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.port != 0
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not parsed.path
            and not any(char.isspace() for char in gateway)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ApiError(
            503,
            "logs_unavailable",
            "Configure TRAINING_SERVICE_RAY_GATEWAY_URL as an HTTPS origin without "
            "credentials, a path, query parameters, or a fragment.",
        )
    if os.getenv("RAY_ADDRESS") or os.getenv("RAY_API_SERVER_ADDRESS"):
        raise ApiError(
            503, "logs_unavailable", "Unset RAY_ADDRESS and RAY_API_SERVER_ADDRESS in the API pod."
        )
    # Give Requests and the WebSocket transport the same explicit trust bundle.
    # A filename also prevents Requests from substituting ambient proxy CA settings.
    ca_file = (
        os.getenv("TRAINING_SERVICE_RAY_CA_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("CURL_CA_BUNDLE")
        or ca_file
        or where()
    )
    try:
        ssl.create_default_context(cafile=ca_file)
    except (OSError, ValueError):
        raise ApiError(
            503,
            "logs_unavailable",
            "The log backend CA bundle cannot be loaded. Check TRAINING_SERVICE_RAY_CA_FILE "
            "and the mounted public CA ConfigMap.",
        ) from None
    return LogBackendConfig(gateway, ca_file)
