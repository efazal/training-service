"""Validate permanent log backend settings without connecting to a training job."""

import os
import ssl
from dataclasses import dataclass
from urllib.parse import urlsplit

from requests.certs import where

from training_service.errors import ApiError


@dataclass(frozen=True)
class LogBackendConfig:
    gateway_url: str
    ca_file: str


def log_backend_config() -> LogBackendConfig:
    """Resolve trust on each new connection so mounted CA updates take effect."""
    gateway = os.getenv("TRAINING_SERVICE_RAY_GATEWAY_URL", "").rstrip("/")
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
