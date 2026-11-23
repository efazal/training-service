"""Caller-scoped, durable MLflow submission reservations; no credential handling."""

from hashlib import sha256
from re import fullmatch
from typing import Any

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from training_service.errors import ApiError
from training_service.kubernetes_client import TIMEOUT

LABEL = "training.rhoai.io/metrics-reservation"
IDENTITY = ("request_digest", "tracking_uri", "experiment_name", "job_name")
FIELDS = {*IDENTITY, "phase", "run_id", "experiment_id", "submit_attempted", "rayjob_uid"}
PHASES = {"prepared", "creating", "reserved", "submitted", "abandoned"}


def _conflict() -> ApiError:
    return ApiError(
        409,
        "conflict",
        "Metrics reservation conflicts with this request or has changed. Reconcile the existing "
        "submission before retrying; use a different job name for a new request.",
    )


def _name(job_name: str) -> str:
    return "training-metrics-" + sha256(job_name.encode()).hexdigest()[:32]


def _field(value: Any, key: str, attribute: str | None = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, value.get(attribute or key))
    return getattr(value, attribute or key, None)


def _read(core: client.CoreV1Api, project: str, job_name: str) -> Any:
    try:
        return core.read_namespaced_config_map(
            name=_name(job_name), namespace=project, _request_timeout=TIMEOUT
        )
    except ApiException as error:
        if error.status != 404:
            raise
        return None


class Reservation:
    """A snapshot whose mutations require the same server UID and resourceVersion."""

    def __init__(self, core: client.CoreV1Api, project: str, job_name: str, resource: Any) -> None:
        self._core, self._project, self._job_name = core, project, job_name
        self._uid = ""
        self._data: dict[str, str] = {}
        self._identity: dict[str, str | None] = {}
        self._take(resource)

    @property
    def submission_id(self) -> str:
        return self._uid

    @property
    def data(self) -> dict[str, str]:
        return self._data.copy()

    def _take(self, resource: Any) -> None:
        meta, data = _field(resource, "metadata"), _field(resource, "data")
        uid, version = _field(meta, "uid"), _field(meta, "resourceVersion", "resource_version")
        labels = _field(meta, "labels")
        if (
            _field(meta, "name") != _name(self._job_name)
            or _field(meta, "namespace") not in (None, self._project)
            or not isinstance(labels, dict)
            or labels.get(LABEL) != "true"
            or not isinstance(uid, str)
            or fullmatch(r"[A-Za-z0-9-]{1,128}", uid) is None
            or not isinstance(version, str)
            or not version
            or not isinstance(data, dict)
            or not set(data) <= FIELDS
            or not all(isinstance(value, str) for value in data.values())
            or ("phase" in data and data["phase"] not in PHASES)
        ):
            raise _conflict()
        identity = {key: data.get(key) for key in IDENTITY}
        if self._uid and (uid != self._uid or identity != self._identity):
            raise _conflict()
        self._uid, self._version, self._identity, self._data = uid, version, identity, data.copy()

    def _patch(self, data: dict[str, str], owners: list[dict[str, str]] | None = None) -> bool:
        meta: dict[str, Any] = {"uid": self._uid, "resourceVersion": self._version}
        if owners is not None:
            meta["ownerReferences"] = owners
        try:
            resource = self._core.patch_namespaced_config_map(
                name=_name(self._job_name),
                namespace=self._project,
                body={"metadata": meta, "data": self._data | data},
                _request_timeout=TIMEOUT,
            )
        except ApiException as error:
            if error.status == 404:
                raise _conflict() from None
            if error.status != 409:
                raise
            self._take(_read(self._core, self._project, self._job_name))
            return False
        self._take(resource)
        return True

    def claim_create(self) -> bool:
        """Claim prepared once; a lost response must be reconciled by submission_id."""
        return self._data.get("phase") == "prepared" and self._patch({"phase": "creating"})

    def update(self, **data_strings: str) -> None:
        if (
            not set(data_strings) <= FIELDS
            or not all(isinstance(value, str) for value in data_strings.values())
            or ("phase" in data_strings and data_strings["phase"] not in PHASES)
        ):
            raise ValueError("Reservation updates require known string data fields and phases.")
        immutable = set(IDENTITY)
        if self._data.get("rayjob_uid"):
            immutable.add("rayjob_uid")
        if any(
            key in immutable and value != self._data.get(key) for key, value in data_strings.items()
        ):
            raise _conflict()
        if data_strings and not self._patch(data_strings):
            raise _conflict()

    def bind_job(self, job: dict[str, Any]) -> None:
        meta = job.get("metadata")
        uid = _field(meta, "uid")
        if (
            not isinstance(uid, str)
            or not uid
            or _field(meta, "name") != self._job_name
            or _field(meta, "namespace") not in (None, self._project)
        ):
            raise _conflict()
        self.update(rayjob_uid=uid)

    def complete(self, job: dict[str, Any]) -> None:
        """Call only after MLflow acknowledges binding to this exact RayJob UID."""
        self.bind_job(job)
        owner = {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "name": self._job_name,
            "uid": self._data["rayjob_uid"],
        }
        if not self._patch({"phase": "submitted"}, [owner]):
            raise _conflict()

    def discard(self) -> None:
        try:
            self._core.delete_namespaced_config_map(
                name=_name(self._job_name),
                namespace=self._project,
                body={"preconditions": {"uid": self._uid}},
                _request_timeout=TIMEOUT,
            )
        except ApiException as error:
            if error.status == 409:
                raise _conflict() from None
            if error.status != 404:
                raise


def acquire(
    core: client.CoreV1Api,
    project: str,
    job_name: str,
    request_digest: str,
    tracking_uri: str,
    experiment_name: str,
) -> Reservation:
    """Create prepared, or validate the existing marker without changing its data."""
    identity = {
        "request_digest": request_digest,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "job_name": job_name,
    }
    body = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": _name(job_name), "labels": {LABEL: "true"}},
        "data": identity | {"phase": "prepared", "submit_attempted": "false"},
    }
    try:
        resource = core.create_namespaced_config_map(
            namespace=project, body=body, _request_timeout=TIMEOUT
        )
    except ApiException as error:
        if error.status != 409:
            raise
        resource = _read(core, project, job_name)
    result = Reservation(core, project, job_name, resource)
    if any(result.data.get(key) != value for key, value in identity.items()):
        raise _conflict()
    if result.data.get("phase") not in PHASES:
        raise _conflict()
    return result


def load(core: client.CoreV1Api, project: str, job_name: str) -> Reservation | None:
    """Load an owned marker; the caller must verify its submission/run/server binding."""
    resource = _read(core, project, job_name)
    return None if resource is None else Reservation(core, project, job_name, resource)
