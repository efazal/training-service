"""One runtime catalog for discovery, readiness, and submission validation."""

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from fastapi import Request
from jsonschema import Draft202012Validator, SchemaError, ValidationError
from referencing import Registry

from training_service.errors import ApiError

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
                                    "api_callable": {"type": "boolean"},
                                    **{
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


@lru_cache(maxsize=1)
def _load_catalog(path: str, image: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise unavailable(
            "startup catalog is missing or invalid; check the discover-algorithms "
            "init-container logs and trainingRuntime.image."
        ) from None
    return validate_catalog(payload, runtime_image=image)


def load_catalog(runtime_image: str | None = None) -> dict[str, Any]:
    """Cache the validated startup file; changing the runtime requires a rollout."""
    path = os.getenv(
        "TRAINING_SERVICE_CATALOG_PATH", "/var/run/training-service/catalog/catalog.json"
    )
    image = os.getenv("TRAINING_RAY_IMAGE", "") if runtime_image is None else runtime_image
    return _load_catalog(path, image)


def read_catalog(request: Request) -> dict[str, Any]:
    del request
    return load_catalog()


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def validate_parameters(
    name: str, parameters: dict[str, Any], backend: str, runtime_image: str | None = None
) -> None:
    """Reject incompatible requests before creating any workload resources."""
    payload = load_catalog(runtime_image)
    algorithm = next((item for item in payload["algorithms"] if item["name"] == name), None)

    def invalid(reason: str) -> ApiError:
        return ApiError(422, "validation_error", reason)

    if not _finite(parameters):
        raise invalid("Algorithm parameters must be finite.")
    if algorithm is None:
        available = ", ".join(item["name"] for item in payload["algorithms"]) or "none"
        raise invalid(
            f"Algorithm '{name}' is not available in the configured training runtime. "
            f"Available algorithms: {available}."
        )
    if backend not in algorithm["supported_backends"]:
        raise invalid(f"Algorithm '{name}' does not support execution backend '{backend}'.")
    schema = algorithm["parameter_schema"]
    metadata = schema.get("x-training-hub", {})
    if metadata.get("unavailable_required_parameters") or metadata.get("api_callable") is False:
        raise invalid(
            f"Algorithm '{name}' requires a Python interface unavailable through this API."
        )
    properties = schema.get("properties", {})
    unknown = sorted(set(parameters) - properties.keys())
    if unknown:
        raise invalid(
            f"Unsupported parameter(s) {', '.join(unknown)} for algorithm '{name}' "
            f"in the configured training runtime. See GET /api/v1/algorithms/{name}."
        )
    # An explicit registry disables jsonschema's default remote-reference retrieval.
    validator = Draft202012Validator(schema, registry=Registry())
    try:
        error = next(validator.iter_errors(parameters), None)
    except Exception:
        raise unavailable(f"cannot validate the parameter schema for algorithm '{name}'.") from None
    if error is None:
        return
    location = ".".join(str(part) for part in error.absolute_path) or "parameters"
    if error.validator == "required":
        missing = sorted(set(error.schema.get("required", [])) - error.instance.keys())
        reason = f"Missing required parameter(s): {', '.join(missing)}."
    elif error.validator in {"enum", "const"}:
        choices = json.dumps(error.validator_value)
        reason = f"Parameter '{location}' must match the runtime's allowed value(s): {choices}."
    else:
        expected: list[str] = []
        for failure in error.context or [error]:
            if failure.validator == "type":
                kinds = failure.validator_value
                expected.extend([kinds] if isinstance(kinds, str) else kinds)
        reason = (
            f"Parameter '{location}' must be {' or '.join(dict.fromkeys(expected))}."
            if expected
            else f"Parameter '{location}' violates the runtime's '{error.validator}' constraint."
        )
    raise invalid(f"Algorithm '{name}': {reason} See GET /api/v1/algorithms/{name}.")
