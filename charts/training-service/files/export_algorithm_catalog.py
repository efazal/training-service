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
from pathlib import Path
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
        if origin is tuple and not (len(arguments) == 2 and arguments[1] is Ellipsis):
            return None
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
    entrypoint = getattr(hub, name, None)
    functions = [algorithm.train]
    if callable(entrypoint):
        # In particular, grpo() has full-parameter semantics which differ from
        # the shared LoRA algorithm class registered under that alias.
        public_parameters = inspect.signature(entrypoint).parameters
        delegates_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in public_parameters.values()
        )
        functions = (
            [algorithm.train, entrypoint] if delegates_kwargs and name != "grpo" else [entrypoint]
        )
    parameters: dict[str, inspect.Parameter] = {}
    for function in functions:
        hints = typing.get_type_hints(function)
        parameters.update(
            {
                key: parameter.replace(annotation=hints.get(key, parameter.annotation))
                for key, parameter in inspect.signature(function).parameters.items()
            }
        )
    backend_schema: dict[str, Any] = {"type": "string", "enum": backends}
    backend_parameter = parameters.get("backend")
    default_backend = backend_parameter.default if backend_parameter is not None else None
    # Report only an explicit callable default, never infer one from registry order.
    if isinstance(default_backend, str) and default_backend in backends:
        backend_schema["default"] = default_backend
    # Registry backend availability does not imply that the public callable
    # accepts a backend keyword (grpo(), for example, fixes its engine internally).
    properties: dict[str, Any] = {"backend": backend_schema} if backend_parameter else {}
    required, unavailable, unavailable_required = [], [], []
    if backend_parameter is not None and backend_parameter.default is inspect.Parameter.empty:
        required.append("backend")
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
            "api_callable": callable(entrypoint),
            "unavailable_callable_parameters": unavailable,
            "unavailable_required_parameters": unavailable_required,
            "managed_parameters": sorted(MANAGED - {"self"}),
        },
    }
    if "default" in backend_schema:
        schema["x-training-hub"]["default_backend"] = backend_schema["default"]
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
    print("Inspecting Training Hub in the configured training runtime", file=sys.stderr)
    with redirect_stdout(sys.stderr):
        import training_hub

        catalog = build_catalog(training_hub, sys.argv[1], version("training-hub"))
    document = json.dumps(catalog, allow_nan=False)
    if len(sys.argv) > 2:
        target = Path(sys.argv[2])
        temporary = target.with_suffix(".tmp")
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(target)
    else:
        print(document)
    print(f"Discovered {len(catalog['algorithms'])} algorithms", file=sys.stderr)


if __name__ == "__main__":
    main()
