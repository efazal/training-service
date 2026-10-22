"""Stateless views and guarded cascading deletion of caller-readable RayJobs."""

import base64
import binascii
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import Request
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from pydantic import ValidationError

from training_service.errors import ApiError
from training_service.kube import TIMEOUT, caller_client, validate_names
from training_service_api.models.create_job_request import CreateJobRequest
from training_service_api.models.resource_spec import ResourceSpec

TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
_STATUSES = TERMINAL | {"PENDING", "RUNNING"}
_ANNOTATION = "training.rhoai.io/"


def read_job(api: client.ApiClient, project: str, name: str) -> dict[str, Any]:
    """Resolve only the requested namespace, using the caller's client."""
    result: dict[str, Any] = client.CustomObjectsApi(api).get_namespaced_custom_object(
        "ray.io", "v1", project, "rayjobs", name, _request_timeout=TIMEOUT
    )
    return result


def job_status(job: dict[str, Any]) -> str:
    """Distinguish infrastructure failure from pending/waiting provisioning."""
    state = job.get("status", {})
    if state.get("jobDeploymentStatus") in {"Failed", "ValidationFailed"}:
        return "FAILED"
    status = state.get("jobStatus", "PENDING")
    return status if status in _STATUSES else "PENDING"


def summary(job: dict[str, Any]) -> dict[str, Any]:
    metadata = job["metadata"]
    result = {
        "id": metadata["name"],
        "status": job_status(job),
        "created_at": metadata["creationTimestamp"],
    }
    if queue := metadata.get("labels", {}).get("kueue.x-k8s.io/queue-name"):
        result["queue_name"] = queue
    return result


def _timestamp(value: Any) -> str | None:
    if isinstance(value, str) and value and not value.startswith("0001-"):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        try:
            return datetime.fromtimestamp(value / 1000, UTC).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    return None


def detail(job: dict[str, Any], project: str) -> dict[str, Any]:
    """Expose allowlisted fields, never raw runtime environments or Secret values."""
    result = summary(job)
    result.update(name=result["id"], project=project, selected_backend="ray")
    annotations = job["metadata"].get("annotations", {})
    if submitter := annotations.get(_ANNOTATION + "submitted-by"):
        result["submitted_by"] = submitter
    submission = annotations.get(_ANNOTATION + "submission")
    if isinstance(submission, str) and len(submission.encode()) <= 65536:
        try:
            accepted = CreateJobRequest.model_validate_json(submission)
            result["submission"] = accepted.model_dump(mode="json", exclude_unset=True)
        except (ValidationError, ValueError):
            # External jobs may have missing, legacy or malformed API annotations.
            pass
    state = job.get("status", {})
    for source, target in (("startTime", "started_at"), ("endTime", "finished_at")):
        if source == "endTime" and result["status"] not in TERMINAL:
            continue
        if timestamp := _timestamp(state.get(source)):
            result[target] = timestamp
    if result["status"] == "FAILED" and state.get("message"):
        result["failure_reason"] = state["message"]
    if resources := _resources(job.get("spec", {}).get("rayClusterSpec", {})):
        result["resources"] = resources
    return result


def _resources(cluster: dict[str, Any]) -> dict[str, Any] | None:
    """Report homogeneous worker configuration; do not invent effective defaults."""
    groups = cluster.get("workerGroupSpecs", [])
    if not groups and cluster.get("headGroupSpec"):
        groups = [dict(cluster["headGroupSpec"], replicas=1)]
    if len(groups) != 1:
        return None
    group = groups[0]
    containers = group.get("template", {}).get("spec", {}).get("containers", [])
    if not containers:
        return None
    requests = containers[0].get("resources", {}).get("requests", {})
    gpu_keys = [key for key in requests if key.endswith("/gpu")]
    if len(gpu_keys) > 1:
        return None
    result: dict[str, Any] = {}
    if isinstance(group.get("replicas"), int) and group["replicas"] > 0:
        result["num_nodes"] = group["replicas"]
    for source, target in (("cpu", "cpus_per_node"), ("memory", "memory_per_node")):
        if source in requests:
            result[target] = str(requests[source])
    if gpu_keys:
        try:
            result.update(gpu_type=gpu_keys[0], gpus_per_node=int(requests[gpu_keys[0]]))
        except (ValueError, TypeError):
            return None
    try:
        ResourceSpec.model_validate(result)
    except ValidationError:
        return None
    return result or None


def _scope(project: str, status: str | None, queue: str | None) -> str:
    return hashlib.sha256(json.dumps([project, status, queue]).encode()).hexdigest()


def _cursor(token: str | None, scope: str) -> str:
    if token is None:
        return ""
    try:
        if len(token) > 2048:
            raise ValueError
        data = json.loads(base64.b64decode(token, altchars=b"-_", validate=True))
        if not isinstance(data, dict) or data.get("scope") != scope:
            raise ValueError
        after = data["after"]
        if not isinstance(after, str) or not after:
            raise ValueError
        return after
    except (ValueError, TypeError, KeyError, binascii.Error):
        raise ApiError(
            400, "invalid_page_token", "Invalid continuation token for these filters."
        ) from None


def list_jobs(
    request: Request,
    project: str,
    status: str | None,
    queue: str | None,
    limit: int | None,
    page_token: str | None,
) -> dict[str, Any]:
    validate_names(project)
    scope = _scope(project, status, queue)
    after = _cursor(page_token, scope)
    items = []
    with caller_client(request) as api:
        # Kubernetes list on a nonexistent namespace may otherwise look empty.
        client.CoreV1Api(api).read_namespace(project, _request_timeout=TIMEOUT)
        custom = client.CustomObjectsApi(api)
        continuation = ""
        while True:
            page = custom.list_namespaced_custom_object(
                "ray.io",
                "v1",
                project,
                "rayjobs",
                limit=500,
                _continue=continuation,
                _request_timeout=TIMEOUT,
            )
            for job in page["items"]:
                item = summary(job)
                if (status is None or item["status"] == status) and (
                    queue is None or item.get("queue_name") == queue
                ):
                    items.append(item)
            continuation = page.get("metadata", {}).get("continue", "")
            if not continuation:
                break
    items.sort(key=lambda item: item["id"])
    remaining = [item for item in items if item["id"] > after]
    selected = remaining[: limit or 20]
    response: dict[str, Any] = {"items": selected, "total_count": len(items)}
    if len(remaining) > len(selected):
        marker = {"scope": scope, "after": selected[-1]["id"]}
        response["next_page_token"] = base64.urlsafe_b64encode(json.dumps(marker).encode()).decode()
    return response


def get_job(request: Request, project: str, name: str) -> dict[str, Any]:
    validate_names(project, name)
    with caller_client(request) as api:
        return detail(read_job(api, project, name), project)


def owned_cluster(api: client.ApiClient, project: str, job: dict[str, Any]) -> str | None:
    """Fail closed rather than act on a shared or caller-injected Ray endpoint."""
    name = job.get("status", {}).get("rayClusterName")
    if not name:
        return None
    validate_names(project, name)
    cluster = client.CustomObjectsApi(api).get_namespaced_custom_object(
        "ray.io", "v1", project, "rayclusters", name, _request_timeout=TIMEOUT
    )
    if not any(
        owner.get("uid") == job["metadata"]["uid"]
        and owner.get("name") == job["metadata"]["name"]
        and owner.get("kind") == "RayJob"
        and owner.get("apiVersion") == "ray.io/v1"
        and owner.get("controller") is True
        for owner in cluster["metadata"].get("ownerReferences", [])
    ):
        raise ApiError(
            409, "unowned_cluster", "RayCluster is not exclusively owned by this RayJob."
        )
    return str(name)


def delete_job(request: Request, project: str, name: str) -> None:
    validate_names(project, name)
    with caller_client(request) as api:
        job = read_job(api, project, name)
        state = job.get("status", {})
        # Use the execution status, not a derived infrastructure-failure label.
        # A retrying deployment can still launch work after an earlier attempt ended.
        if state.get("jobStatus") not in TERMINAL or state.get("jobDeploymentStatus") in {
            "Retrying",
            "Initializing",
            "Waiting",
            "Suspending",
            "Suspended",
        }:
            raise ApiError(409, "job_not_terminal", "Only terminal RayJobs can be deleted.")
        if job.get("spec", {}).get("clusterSelector"):
            raise ApiError(
                409, "shared_cluster", "Shared RayClusters cannot be deleted by this API."
            )
        try:
            owned_cluster(api, project, job)
        except ApiException as error:
            if error.status != 404:
                raise
            # Normal when shutdown-after-finish already removed the owned cluster.
        metadata = job["metadata"]
        client.CustomObjectsApi(api).delete_namespaced_custom_object(
            "ray.io",
            "v1",
            project,
            "rayjobs",
            name,
            body=client.V1DeleteOptions(
                propagation_policy="Foreground",
                preconditions=client.V1Preconditions(
                    uid=metadata["uid"],
                    resource_version=metadata["resourceVersion"],
                ),
            ),
            _request_timeout=TIMEOUT,
        )
