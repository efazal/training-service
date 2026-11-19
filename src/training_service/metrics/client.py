"""Small MLflow REST client shared by the API and the training runtime.

Only standard-library imports: training images need no extra tracking packages.
Credentials are supplied per request and are never included in portable metadata.
"""

from __future__ import annotations

import json
import re
import ssl
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener


class TrackingError(Exception):
    """Sanitized transport error, without server responses or request headers."""

    def __init__(self, status: int = 503) -> None:
        self.status = status
        super().__init__(f"MLflow request failed (HTTP {status}).")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def valid_url(value: str) -> bool:
    if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() for c in value):
        return False
    try:
        url = urlsplit(value)
        return bool(
            url.scheme == "https"
            and url.hostname
            and url.port != 0
            and not url.username
            and not url.password
            and not url.query
            and not url.fragment
        )
    except ValueError:
        return False


class TrackingClient:
    """An isolated client; workspace and authentication never use global defaults."""

    def __init__(
        self,
        uri: str,
        project: str,
        token: Callable[[], str],
        *,
        ca_file: str | None = None,
        ca_bundle: str | None = None,
    ) -> None:
        if not valid_url(uri) or not re.fullmatch(r"[a-z0-9][-a-z0-9]{0,62}", project):
            raise TrackingError()
        self.uri = uri.rstrip("/")
        self.project = project
        self._token = token
        context = ssl.create_default_context(cafile=ca_file)
        if ca_bundle:
            context.load_verify_locations(cadata=ca_bundle)
        self._opener = build_opener(_NoRedirect(), HTTPSHandler(context=context))

    def request(self, method: str, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        url = self.uri + "/api/2.0/mlflow/" + endpoint
        body = None
        if method == "GET":
            url += "?" + urlencode(data)
        else:
            body = json.dumps(data, allow_nan=False).encode()
        try:
            request = Request(
                url,
                data=body,
                method=method,
                headers={
                    "Authorization": "Bearer " + self._token(),
                    "X-MLFLOW-WORKSPACE": self.project,
                    "Content-Type": "application/json",
                },
            )
            with self._opener.open(request, timeout=5) as response:
                payload = response.read(2**20 + 1)
            if len(payload) > 2**20:
                raise TrackingError()
            result = json.loads(payload or b"{}")
            if not isinstance(result, dict):
                raise TrackingError()
            return result
        except HTTPError as error:
            error.close()
            raise TrackingError(error.code) from None
        except (URLError, OSError, ValueError, UnicodeError):
            raise TrackingError() from None

    def experiment(self, name: str) -> str:
        try:
            result = self.request("GET", "experiments/get-by-name", {"experiment_name": name})
            return str(result["experiment"]["experiment_id"])
        except TrackingError as error:
            if error.status != 404:
                raise
        try:
            return str(self.request("POST", "experiments/create", {"name": name})["experiment_id"])
        except TrackingError as error:
            if error.status not in {400, 409}:
                raise
            # Another submission can create the same experiment concurrently.
            result = self.request("GET", "experiments/get-by-name", {"experiment_name": name})
            return str(result["experiment"]["experiment_id"])

    def create_run(self, experiment: str, name: str, tags: dict[str, str]) -> str:
        result = self.request(
            "POST",
            "runs/create",
            {
                "experiment_id": experiment,
                "run_name": name,
                "start_time": int(time.time() * 1000),
                "tags": [{"key": key, "value": value} for key, value in tags.items()],
            },
        )
        identifier = result["run"]["info"]["run_id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise TrackingError()
        return identifier

    def log(
        self,
        run_id: str,
        *,
        metrics: dict[str, float] | None = None,
        params: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        step: int = 0,
    ) -> None:
        now = int(time.time() * 1000)
        self.request(
            "POST",
            "runs/log-batch",
            {
                "run_id": run_id,
                "metrics": [
                    {"key": key, "value": value, "timestamp": now, "step": step}
                    for key, value in (metrics or {}).items()
                ],
                "params": [{"key": key, "value": value} for key, value in (params or {}).items()],
                "tags": [{"key": key, "value": value} for key, value in (tags or {}).items()],
            },
        )

    def finish(self, run_id: str, status: str) -> None:
        self.request(
            "POST",
            "runs/update",
            {"run_id": run_id, "status": status, "end_time": int(time.time() * 1000)},
        )
