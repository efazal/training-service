"""Validate permanent log backend settings without connecting to a training job."""

import logging
import os
import re
import ssl
import threading
from dataclasses import dataclass
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
            "gateway discovery will retry on the next readiness check.",
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


class LogBackendResolver:
    """Pin the first valid gateway per backend; failed discovery remains retryable."""

    def __init__(self) -> None:
        self._gateway: str | None = None
        self._lock = threading.Lock()

    def configuration(self) -> "LogBackendConfig":
        with self._lock:
            if self._gateway is None:
                override = os.getenv("TRAINING_SERVICE_RAY_GATEWAY_URL")
                gateway = override or _discover_gateway()
                resolved = log_backend_config(gateway)
                self._gateway = resolved.gateway_url
                source = (
                    "TRAINING_SERVICE_RAY_GATEWAY_URL"
                    if override
                    else "GatewayConfig/default-gateway status.domain"
                )
                _LOG.info("Resolved log gateway %s from %s", self._gateway, source)
                return resolved
            gateway = self._gateway
        # Trust files can rotate independently of the pinned gateway address.
        return log_backend_config(gateway)


@dataclass(frozen=True)
class LogBackendConfig:
    gateway_url: str
    ca_file: str


def log_backend_config(gateway_url: str | None = None) -> LogBackendConfig:
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
