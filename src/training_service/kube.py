"""Isolated caller-authenticated Kubernetes access for job operations."""

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import Request
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError

from training_service.auth import CallerIdentity
from training_service.errors import ApiError

TIMEOUT = (5, 15)


def validate_names(project: str, job_id: str | None = None) -> None:
    """Reject path-like values before constructing Kubernetes or gateway paths."""
    label = r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
    if len(project) > 63 or re.fullmatch(label, project) is None:
        raise ApiError(400, "bad_request", "Invalid OpenShift project name.")
    if job_id is not None and (
        len(job_id) > 253 or re.fullmatch(rf"{label}(?:\.{label})*", job_id) is None
    ):
        raise ApiError(400, "bad_request", "Invalid RayJob name.")


@contextmanager
def caller_client(request: Request) -> Iterator[client.ApiClient]:
    """Forward only a verified caller's credential; never mutate SDK defaults."""
    if not isinstance(getattr(request.state, "caller_identity", None), CallerIdentity):
        raise ApiError(403, "forbidden", "Verified caller context is unavailable.")
    configuration = client.Configuration()
    configuration.host = os.getenv(
        "TRAINING_SERVICE_KUBERNETES_API_URL", "https://kubernetes.default.svc"
    )
    configuration.ssl_ca_cert = os.getenv(
        "TRAINING_SERVICE_KUBERNETES_CA_FILE",
        "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    )
    configuration.verify_ssl = True
    configuration.api_key["authorization"] = request.headers["authorization"]
    configuration.retries = 0
    try:
        with client.ApiClient(configuration) as api:
            yield api
    except ApiException as error:
        codes = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found"}
        codes[409] = "conflict"
        status = error.status if error.status in codes else 503
        raise ApiError(
            status,
            codes.get(status, "backend_unavailable"),
            "The requested Kubernetes operation could not be completed.",
        ) from None
    except (HTTPError, OSError):
        raise ApiError(
            503, "backend_unavailable", "Kubernetes is temporarily unavailable."
        ) from None
