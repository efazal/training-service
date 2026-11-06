"""Human-facing views of the runtime catalog; submission keeps the raw schemas."""

import re
from copy import deepcopy
from typing import Any
from urllib.parse import quote

from fastapi import Request
from jsonschema import Draft202012Validator

from training_service import catalog
from training_service.errors import ApiError

_SUMMARIES = {
    "lora_sft": "Supervised fine-tuning using LoRA adapters.",
    "sft": "Full-parameter supervised fine-tuning.",
    "osft": "Orthogonal subspace fine-tuning to help preserve existing model capabilities.",
    "lora_grpo": "LoRA fine-tuning using Group Relative Policy Optimization.",
    "grpo": "Full-parameter fine-tuning using Group Relative Policy Optimization.",
}
_PARAMETER_DESCRIPTIONS = {
    "backend": "Training Hub engine; distinct from the job execution backend.",
    "learning_rate": "Learning rate used during training.",
    "num_epochs": "Number of passes over the training dataset.",
    "effective_batch_size": (
        "Total samples per optimizer update across GPUs and gradient accumulation."
    ),
    "max_seq_len": "Maximum number of tokens per training sample.",
}


def _short_description(text: str) -> str:
    """Use a bounded first paragraph, never a full Python docstring."""
    paragraph = " ".join(text.strip().split("\n\n", 1)[0].split())
    if len(paragraph) <= 240:
        return paragraph
    return paragraph[:237].rsplit(" ", 1)[0] + "..."


def _summary(algorithm: dict[str, Any]) -> dict[str, Any]:
    name = algorithm["name"]
    return {
        "name": name,
        "description": _SUMMARIES.get(name)
        or _short_description(algorithm.get("description", name)),
        "supported_backends": algorithm["supported_backends"],
        "details_url": "/api/v1/algorithms/" + quote(name, safe=""),
    }


def list_algorithms(request: Request) -> dict[str, Any]:
    """Return only the small catalog view; never mutate the validation artifact."""
    payload = catalog.read_catalog(request)
    result: dict[str, Any] = {"algorithms": [_summary(item) for item in payload["algorithms"]]}
    if "runtime" in payload:
        result["runtime"] = deepcopy(payload["runtime"])
    return result


def _parameter_docs(docstring: str) -> dict[str, str]:
    """Extract concise argument descriptions from Training Hub's Google-style docs."""
    result = {}
    for line in docstring.splitlines():
        match = re.match(r"\s{4,}(\w+)(?:\s*\([^)]*\))?:\s+(.+)", line)
        if match:
            result[match[1]] = _short_description(match[2])
    return result


def _types(schema: dict[str, Any] | bool) -> list[str]:
    if isinstance(schema, bool):
        return []
    value = schema.get("type", [])
    result = [value] if isinstance(value, str) else list(value)
    for branch in schema.get("anyOf", []) + schema.get("oneOf", []):
        result.extend(_types(branch))
    return list(dict.fromkeys(result))


def _has_references(value: Any) -> bool:
    """Do not resolve catalog references or fetch remote schemas to validate examples."""
    if isinstance(value, dict):
        return (
            "$ref" in value
            or "$dynamicRef" in value
            or any(_has_references(v) for v in value.values())
        )
    return isinstance(value, list) and any(_has_references(v) for v in value)


def _parameter(schema: dict[str, Any] | bool, name: str, docs: dict[str, str]) -> dict[str, Any]:
    if isinstance(schema, bool):
        schema = {} if schema else {"not": {}}
    types = _types(schema)
    result: dict[str, Any] = {
        "type": " | ".join(value for value in types if value != "null") or "any",
        "description": _PARAMETER_DESCRIPTIONS.get(name)
        or _short_description(schema.get("description") or docs.get(name, ""))
        or name.replace("_", " ").capitalize() + ".",
    }
    if "null" in types:
        result["nullable"] = True
    # Native None defaults can delegate the effective default to the backend;
    # they must not be presented as a concrete training default.
    if schema.get("default") is not None:
        result["default"] = schema["default"]
    if "enum" in schema:
        result["choices"] = schema["enum"]
    if "const" in schema:
        result["choices"] = [schema["const"]]
    candidate = schema.get("example")
    if (
        candidate is not None
        and not _has_references(schema)
        and Draft202012Validator(schema).is_valid(candidate)
    ):
        result["example"] = candidate
    return result


def get_algorithm(request: Request, name: str, include: str | None = None) -> dict[str, Any]:
    """Return readable details, with the exact validation schema only on opt-in."""
    if include not in (None, "parameter_schema"):
        raise ApiError(400, "bad_request", "include must be parameter_schema when supplied.")
    payload = catalog.read_catalog(request)
    algorithm = next((item for item in payload["algorithms"] if item["name"] == name), None)
    if algorithm is None:
        raise ApiError(
            404, "not_found", f"Algorithm '{name}' was not found in the configured catalog."
        )
    schema = algorithm["parameter_schema"]
    docs = _parameter_docs(algorithm.get("description", ""))
    result = _summary(algorithm)
    result.update(
        required_parameters=list(schema.get("required", [])),
        parameters={
            key: _parameter(value, key, docs) for key, value in schema.get("properties", {}).items()
        },
        parameter_schema_url=result["details_url"] + "?include=parameter_schema",
    )
    notes = [
        "Omitted parameters use runtime/backend defaults. Parameter examples, when supplied "
        "by the runtime schema, are illustrative and are not defaults.",
        "Submission validates algorithms and parameters against this runtime catalog. "
        "A supported execution profile is also required; runtime availability alone does "
        "not enable submission. Python-only interfaces and non-JSON parameters are not supported.",
    ]
    metadata = schema.get("x-training-hub", {})
    if metadata.get("unavailable_required_parameters") or metadata.get("api_callable") is False:
        notes.append(
            "This algorithm requires a Python interface and cannot be submitted through this API."
        )
    if metadata.get("unavailable_callable_parameters") or metadata.get(
        "unavailable_required_parameters"
    ):
        notes.append(
            "Python callbacks and custom rollout functions cannot be supplied as JSON. "
            "Only JSON-configurable training modes are available through this API."
        )
    result["notes"] = notes
    if "runtime" in payload:
        result["runtime"] = deepcopy(payload["runtime"])
    if include == "parameter_schema":
        result["parameter_schema"] = deepcopy(schema)
    return result
