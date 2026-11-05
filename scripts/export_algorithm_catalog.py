"""Inspect Training Hub in the workload image, without an API client or GPU.

This module uses only the standard library until discovery imports Training Hub.
Run this script inside the exact training runtime image; stdout is catalog JSON.
"""

import collections.abc
import hashlib
import inspect
import json
import sys
import types
import typing
from contextlib import redirect_stdout
from importlib.metadata import version
from typing import Any, get_args, get_origin

# These values belong to the API's storage, security, or resource adapters, not
# to user-supplied training hyperparameters. Unknown **kwargs are not accepted.
MANAGED = {
    "self",
    "model_path",
    "data_path",
    "ckpt_output_dir",
    "data_output_dir",
    "nproc_per_node",
    "nnodes",
    "n_gpus",
    "node_rank",
    "rdzv_id",
    "rdzv_endpoint",
    "master_addr",
    "master_port",
    "art_path",
    "checkpoint_storage",
}


def json_type(annotation: Any) -> dict[str, Any] | None:
    """Represent JSON-compatible annotations; never turn callables into code."""
    if annotation in (Any, inspect.Parameter.empty):
        return {}
    if annotation is None or annotation is type(None):
        return {"type": "null"}
    primitive = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if annotation in primitive:
        return {"type": primitive[annotation]}
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        schemas = [json_type(argument) for argument in arguments]
        if any(schema is None for schema in schemas):
            return None
        return {"anyOf": schemas}
    if origin is typing.Literal:
        return {"enum": list(arguments)}
    if origin in (list, tuple, collections.abc.Sequence) or annotation is list:
        items = json_type(arguments[0]) if arguments else {}
        return {"type": "array", "items": items} if items is not None else None
    if origin is dict or annotation is dict:
        values = json_type(arguments[1]) if len(arguments) == 2 else {}
        return {"type": "object", "additionalProperties": values} if values is not None else None
    return None


def fingerprint(hub: Any) -> str:
    """Detect registry/source drift between discovery and a submitted job."""
    registry = hub.AlgorithmRegistry
    source = []
    for name in sorted(registry.list_algorithms()):
        source.append(name + inspect.getsource(registry.get_algorithm(name)))
        for backend in sorted(registry.list_backends(name)):
            source.append(backend + inspect.getsource(registry.get_backend(name, backend)))
    return hashlib.sha256("\n".join(source).encode()).hexdigest()


def describe(hub: Any, name: str) -> dict[str, Any]:
    registry = hub.AlgorithmRegistry
    algorithm = registry.get_algorithm(name)
    backends = registry.list_backends(name)
    if not backends:
        raise ValueError(f"Training Hub algorithm {name} has no registered backend")
    # Ray's bundled GRPO example uses verl, not ART's co-located local engine.
    default_backend = "verl" if "verl" in backends else backends[0]
    entrypoint = getattr(hub, name, None)
    functions = [algorithm.train]
    if callable(entrypoint):
        # In particular, grpo() has full-parameter semantics which differ from
        # the shared LoRA algorithm class registered under that alias.
        functions = [entrypoint] if name == "grpo" else [algorithm.train, entrypoint]
    parameters: dict[str, inspect.Parameter] = {}
    for function in functions:
        hints = typing.get_type_hints(function)
        parameters.update(
            {
                key: parameter.replace(annotation=hints.get(key, parameter.annotation))
                for key, parameter in inspect.signature(function).parameters.items()
            }
        )
    properties: dict[str, Any] = {
        "backend": {"type": "string", "enum": backends, "default": default_backend}
    }
    required, unavailable, unavailable_required = [], [], []
    for key, parameter in parameters.items():
        if (
            key in MANAGED
            or key == "backend"
            or parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ):
            continue
        schema = json_type(parameter.annotation)
        if schema is None:
            unavailable.append(key)
            if parameter.default is inspect.Parameter.empty:
                unavailable_required.append(key)
            continue
        if key == "trust_remote_code":
            schema = {"type": "boolean", "const": False}
        if parameter.default is not inspect.Parameter.empty:
            if parameter.default is None or isinstance(parameter.default, (str, int, float, bool)):
                schema["default"] = parameter.default
        else:
            required.append(key)
        if key == "trust_remote_code":
            schema["default"] = False
        properties[key] = schema
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        "x-training-hub": {
            "default_backend": default_backend,
            "unavailable_callable_parameters": unavailable,
            "unavailable_required_parameters": unavailable_required,
            "managed_parameters": sorted(MANAGED - {"self"}),
        },
    }
    return {
        "name": name,
        "description": inspect.getdoc(entrypoint or algorithm) or name,
        "supported_backends": ["ray"],
        "parameter_schema": schema,
    }


def build_catalog(hub: Any, image: str, hub_version: str) -> dict[str, Any]:
    return {
        "runtime": {
            "image": image,
            "training_hub_version": hub_version,
            "registry_sha256": fingerprint(hub),
        },
        "algorithms": [
            describe(hub, name) for name in sorted(hub.AlgorithmRegistry.list_algorithms())
        ],
    }


def main() -> None:
    # Some registry imports print banners. Keep stdout machine-readable.
    with redirect_stdout(sys.stderr):
        import training_hub

        catalog = build_catalog(training_hub, sys.argv[1], version("training-hub"))
    print(json.dumps(catalog))


if __name__ == "__main__":
    main()
