"""Isolated caller-authenticated Kubernetes access for job operations."""

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit

from fastapi import Request
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError

from training_service.auth import CallerIdentity
from training_service.errors import ApiError

TIMEOUT = (5, 15)


@contextmanager
def describe_not_found(resource: str, name: str, *, project: str | None = None) -> Iterator[None]:
    """Describe a known resource lookup without trusting Kubernetes response bodies."""
    try:
        yield
    except ApiException as error:
        if error.status != 404:
            raise
        message = f"{resource} '{name}' was not found"
        if project is not None:
            message += f" in project '{project}'"
        raise ApiError(404, "not_found", message + ".") from None


def validate_names(project: str, job_id: str | None = None) -> None:
    """Reject path-like values before constructing Kubernetes or gateway paths."""
    label = r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
    if len(project) > 63 or re.fullmatch(label, project) is None:
        raise ApiError(400, "bad_request", "Invalid OpenShift project name.")
    if job_id is not None and (
        len(job_id) > 253 or re.fullmatch(rf"{label}(?:\.{label})*", job_id) is None
    ):
        raise ApiError(400, "bad_request", "Invalid RayJob name.")


def caller_configuration(authorization: str) -> client.Configuration:
    """Build isolated workload configuration without loading service-account tokens."""
    # Do not use the in-cluster loader: its credential refresh hook could replace
    # a caller's token with the pod's service-account token.
    host = os.getenv("TRAINING_SERVICE_KUBERNETES_API_URL", "https://kubernetes.default.svc")
    try:
        parsed = urlsplit(host)
        valid = (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.port != 0
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ApiError(
            503,
            "backend_unavailable",
            "Configure TRAINING_SERVICE_KUBERNETES_API_URL as an HTTPS URL "
            "without credentials, query parameters, or a fragment.",
        )
    configuration = client.Configuration()
    configuration.host = host
    configuration.ssl_ca_cert = os.getenv(
        "TRAINING_SERVICE_KUBERNETES_CA_FILE",
        "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    )
    configuration.verify_ssl = True
    configuration.api_key["authorization"] = authorization
    configuration.api_key_prefix.pop("authorization", None)
    configuration.retries = 0
    return configuration


@contextmanager
def caller_client(request: Request) -> Iterator[client.ApiClient]:
    """Forward only a verified caller's credential; never mutate SDK defaults."""
    if not isinstance(getattr(request.state, "caller_identity", None), CallerIdentity):
        raise ApiError(403, "forbidden", "Verified caller context is unavailable.")
    configuration = caller_configuration(request.headers["authorization"])
    try:
        with client.ApiClient(configuration) as api:
            yield api
    except ApiException as error:
        codes = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found"}
        codes[409] = "conflict"
        status = error.status if error.status in codes else 503
        messages = {
            400: "Kubernetes rejected the operation. Check the request parameters.",
            401: "The OpenShift bearer token is no longer valid. Sign in again and retry.",
            403: "Caller is not permitted to access the requested project resource.",
            404: "The requested Kubernetes resource was not found.",
            409: "The operation conflicts with the current resource state. Refresh and retry.",
            503: "Kubernetes is temporarily unavailable. Try again later.",
        }
        raise ApiError(
            status,
            codes.get(status, "backend_unavailable"),
            messages[status],
            headers={"WWW-Authenticate": "Bearer"} if status == 401 else None,
        ) from None
    except (HTTPError, OSError):
        raise ApiError(
            503, "backend_unavailable", "Kubernetes is temporarily unavailable."
        ) from None
