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
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener


class TrackingError(Exception):
    """Sanitized transport error, without server responses or request headers."""

    def __init__(self, status: int = 503) -> None:
        self.status = status
        super().__init__(f"MLflow request failed (HTTP {status}).")


class DeletedExperimentError(TrackingError):
    """An experiment must be restored explicitly before accepting new runs."""


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
            and "?" not in value
            and "#" not in value
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
            # Scale with payload so large metric batches are not cut off at 5s.
            timeout = 5 + (len(body) if body else 0) / 200_000
            with self._opener.open(request, timeout=timeout) as response:
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
        except (HTTPException, URLError, OSError, ValueError, UnicodeError):
            raise TrackingError() from None

    @staticmethod
    def experiment_id(result: dict[str, Any]) -> str:
        value = result.get("experiment_id")
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
            raise TrackingError()
        return value

    def _experiment_id(self, result: dict[str, Any]) -> str:
        experiment = result.get("experiment")
        if not isinstance(experiment, dict):
            raise TrackingError()
        if experiment.get("lifecycle_stage") == "deleted":
            raise DeletedExperimentError(409)
        return self.experiment_id(experiment)

    def experiment(self, name: str) -> str:
        try:
            return self._experiment_id(
                self.request("GET", "experiments/get-by-name", {"experiment_name": name})
            )
        except TrackingError as error:
            if error.status != 404:
                raise
        try:
            return self.experiment_id(self.request("POST", "experiments/create", {"name": name}))
        except TrackingError as error:
            if error.status not in {400, 409}:
                raise
            # Another submission can create the same experiment concurrently.
            return self._experiment_id(
                self.request("GET", "experiments/get-by-name", {"experiment_name": name})
            )

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.request("GET", "runs/get", {"run_id": run_id}).get("run")
        if not isinstance(run, dict) or not isinstance(run.get("info"), dict):
            raise TrackingError()
        info = run["info"]
        if not isinstance(info.get("status"), str):
            raise TrackingError()
        if info.get("run_id") != run_id or info.get("status") not in {
            "RUNNING",
            "SCHEDULED",
            "FINISHED",
            "FAILED",
            "KILLED",
        }:
            raise TrackingError()
        self.experiment_id(info)
        data = run.get("data", {})
        if not isinstance(data, dict):
            raise TrackingError()
        for key in ("tags", "params"):
            values = data.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(item, dict)
                or not isinstance(item.get("key"), str)
                or not isinstance(item.get("value"), str)
                for item in values
            ):
                raise TrackingError()
        return run

    def create_run(self, experiment: str, name: str, tags: dict[str, str]) -> str:
        try:
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
            run = result.get("run")
            info = run.get("info") if isinstance(run, dict) else None
            identifier = info.get("run_id") if isinstance(info, dict) else None
            if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
                raise TrackingError()
            return identifier
        except TrackingError as error:
            submission_id = tags.get("training.submission_id")
            if submission_id and (error.status >= 500 or error.status in {408, 429}):
                recovered = self.find_run(experiment, submission_id)
                if recovered:
                    return recovered
            raise

    def find_run(self, experiment: str, submission_id: str) -> str | None:
        """Reconcile only a server-issued reservation identity in this workspace."""
        if not re.fullmatch(r"[a-zA-Z0-9-]{1,128}", submission_id):
            raise TrackingError()
        result = self.request(
            "POST",
            "runs/search",
            {
                "experiment_ids": [experiment],
                "filter": f"tags.`training.submission_id` = '{submission_id}'",
                "max_results": 2,
                "run_view_type": "ALL",
            },
        )
        runs = result.get("runs", [])
        if not isinstance(runs, list) or len(runs) > 1 or result.get("next_page_token"):
            raise TrackingError()
        if not runs:
            return None
        run = runs[0]
        if not isinstance(run, dict) or not isinstance(run.get("info"), dict):
            raise TrackingError()
        identifier = run["info"].get("run_id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[a-f0-9]{32}", identifier)
            or run["info"].get("experiment_id") != experiment
        ):
            raise TrackingError()
        data = run.get("data")
        values = data.get("tags") if isinstance(data, dict) else None
        if not isinstance(values, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("key"), str)
            or not isinstance(item.get("value"), str)
            for item in values
        ):
            raise TrackingError()
        tags = {item["key"]: item["value"] for item in values}
        if (
            tags.get("training.submission_id") != submission_id
            or tags.get("training.project") != self.project
            or run["info"].get("lifecycle_stage") == "deleted"
        ):
            raise TrackingError()
        return identifier

    def log_batch(self, run_id: str, metrics: list[dict[str, Any]], tags: dict[str, str]) -> None:
        # Preserve every point's original timestamp and step, including repeated keys.
        if len(tags) > 100:
            raise TrackingError()
        labels = [{"key": key, "value": value} for key, value in tags.items()]
        body = {"run_id": run_id, "metrics": [], "params": [], "tags": labels}
        try:
            overhead = len(json.dumps(body, allow_nan=False).encode())
            sizes = [len(json.dumps(point, allow_nan=False).encode()) for point in metrics]
        except (TypeError, ValueError, OverflowError):
            raise TrackingError() from None
        # The REST limit applies to the encoded JSON, including Unicode escapes.
        # Reserve the final tags in every chunk's budget so the last one always fits.
        maximum = 1_000_000
        if overhead > maximum:
            raise TrackingError(413)
        chunks: list[list[dict[str, Any]]] = []
        chunk: list[dict[str, Any]] = []
        size = overhead
        for point, length in zip(metrics, sizes, strict=True):
            if overhead + length > maximum:
                raise TrackingError(413)
            if len(chunk) >= 1000 - len(tags) or size + length + (2 if chunk else 0) > maximum:
                chunks.append(chunk)
                chunk, size = [], overhead
            size += length + (2 if chunk else 0)
            chunk.append(point)
        chunks.append(chunk)
        for index, points in enumerate(chunks):
            self.request(
                "POST",
                "runs/log-batch",
                {
                    "run_id": run_id,
                    "metrics": points,
                    "params": [],
                    "tags": labels if index == len(chunks) - 1 else [],
                },
            )

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
        terminal = {"FINISHED", "FAILED", "KILLED"}
        if status not in terminal:
            raise TrackingError()
        run = self.get_run(run_id)
        if run["info"]["status"] in terminal:
            return
        # MLflow parameters are immutable. Both the driver and DELETE claim the
        # same parameter before updating status, so competing writers converge
        # on one result without overwriting an earlier completion with KILLED.
        result: dict[str, Any] = {"status": status, "end_time": int(time.time() * 1000)}
        try:
            self.log(run_id, params={"training.terminal_result": json.dumps(result)})
        except TrackingError:
            # A conflicting claim or a lost response may still have committed.
            run = self.get_run(run_id)
            if run["info"]["status"] in terminal:
                return
            params = {item["key"]: item["value"] for item in run.get("data", {}).get("params", [])}
            try:
                result = json.loads(params["training.terminal_result"])
                if (
                    not isinstance(result, dict)
                    or result.get("status") not in terminal
                    or type(result.get("end_time")) is not int
                    or result["end_time"] < 0
                ):
                    raise ValueError("Invalid terminal result")
            except (KeyError, ValueError, TypeError):
                raise TrackingError() from None
        self.request(
            "POST",
            "runs/update",
            {"run_id": run_id, "status": result["status"], "end_time": result["end_time"]},
        )

    def delete_run(self, run_id: str) -> None:
        self.request("POST", "runs/delete", {"run_id": run_id})
