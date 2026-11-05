"""Read the configured algorithm catalog directly with the caller's K8s identity."""

import json
import os
from typing import Any, cast

from fastapi import Request
from jsonschema import Draft202012Validator, SchemaError, ValidationError
from kubernetes import client

from training_service.errors import ApiError
from training_service.kubernetes_client import TIMEOUT, caller_client

_CATALOG_SCHEMA = {
    "type": "object",
    "required": ["algorithms"],
    "properties": {
        "algorithms": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "supported_backends", "parameter_schema"],
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
                    "description": {"type": "string"},
                    "supported_backends": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "parameter_schema": {
                        "type": "object",
                        "properties": {
                            "properties": {
                                "type": "object",
                                "additionalProperties": {"type": ["object", "boolean"]},
                            },
                            "x-training-hub": {
                                "type": "object",
                                "properties": {
                                    key: {"type": "array", "items": {"type": "string"}}
                                    for key in (
                                        "unavailable_required_parameters",
                                        "unavailable_callable_parameters",
                                    )
                                },
                            },
                        },
                    },
                },
            },
        },
        "runtime": {
            "type": "object",
            "required": ["image", "training_hub_version", "registry_sha256"],
            "properties": {
                "image": {"type": "string", "minLength": 1},
                "training_hub_version": {"type": "string", "minLength": 1},
                "registry_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
    },
}


def unavailable(reason: str) -> ApiError:
    return ApiError(503, "backend_unavailable", "Algorithm catalog unavailable: " + reason)


def validate_catalog(payload: Any, runtime_image: str | None = None) -> dict[str, Any]:
    """Reject malformed descriptors before exposing them through the API."""
    try:
        Draft202012Validator(_CATALOG_SCHEMA).validate(payload)
        names = set()
        for algorithm in payload["algorithms"]:
            if algorithm["name"] in names:
                raise ValueError("Duplicate algorithm name")
            names.add(algorithm["name"])
            Draft202012Validator.check_schema(algorithm["parameter_schema"])
        runtime = payload.get("runtime")
        image = os.getenv("TRAINING_RAY_IMAGE") if runtime_image is None else runtime_image
        if image and payload["algorithms"] and not runtime:
            raise ValueError("Runtime provenance missing")
        if runtime and image and runtime["image"] != image:
            raise ValueError("Runtime image mismatch")
    except (KeyError, TypeError, ValueError, ValidationError, SchemaError):
        raise unavailable(
            "configured catalog is malformed or does not match the runtime image."
        ) from None
    return cast(dict[str, Any], payload)


def read_catalog(request: Request) -> dict[str, Any]:
    namespace = os.getenv("TRAINING_SERVICE_CATALOG_NAMESPACE", "")
    name = os.getenv("TRAINING_SERVICE_CATALOG_NAME", "")
    if not namespace or not name:
        raise unavailable(
            "configure TRAINING_SERVICE_CATALOG_NAMESPACE and TRAINING_SERVICE_CATALOG_NAME."
        )
    try:
        with caller_client(request) as api:
            configmap = client.CoreV1Api(api).read_namespaced_config_map(
                name=name, namespace=namespace, _request_timeout=TIMEOUT
            )
    except ApiError as error:
        if error.status_code == 401:
            raise
        if error.status_code == 403:
            raise ApiError(
                403,
                "forbidden",
                f"Caller cannot read algorithm catalog ConfigMap '{name}' "
                f"in project '{namespace}'.",
            ) from None
        raise unavailable(f"cannot read ConfigMap '{name}' in project '{namespace}'.") from None
    try:
        payload = json.loads((configmap.data or {})["catalog.json"])
    except (KeyError, TypeError, ValueError):
        raise unavailable(f"ConfigMap '{name}' must contain valid JSON in catalog.json.") from None
    return validate_catalog(payload)
