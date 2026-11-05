"""No Training Hub, GPU or cluster needed to verify registry introspection."""

import json
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from training_service.catalog import validate_catalog

EXPORTER = (
    Path(__file__).resolve().parents[2]
    / "charts/training-service/files/export_algorithm_catalog.py"
)


class Algorithm:
    def train(
        self,
        model_path: str,
        data_path: str,
        learning_rate: float,
        enabled: bool = False,
        count: int = 0,
        reward_fn: Callable | None = None,
    ):
        """A future algorithm.\n\nArgs:\n    learning_rate: Training rate."""


class Registry:
    @staticmethod
    def list_algorithms():
        return ["future_algorithm", "grpo"]

    @staticmethod
    def get_algorithm(name):
        return Algorithm

    @staticmethod
    def list_backends(name):
        return ["engine"]

    @staticmethod
    def get_backend(name, backend):
        return Algorithm


@pytest.fixture
def exported():
    exporter = runpy.run_path(str(EXPORTER))

    def grpo(model_path: str, data_path: str, num_iterations: int = 15, backend: str = "engine"):
        """Full-parameter GRPO, without shared LoRA parameters."""

    hub = SimpleNamespace(AlgorithmRegistry=Registry, grpo=grpo)
    return exporter["build_catalog"](hub, "unit-runtime", "unit-version")


def test_export_uses_registry_and_preserves_schemas(exported):
    assert validate_catalog(exported) == exported
    assert exported["runtime"]["image"] == "unit-runtime"
    assert len(exported["runtime"]["registry_sha256"]) == 64
    algorithm = exported["algorithms"][0]
    assert algorithm["name"] == "future_algorithm"
    schema = algorithm["parameter_schema"]
    assert schema["required"] == ["learning_rate"]
    assert schema["properties"]["enabled"]["default"] is False
    assert schema["properties"]["count"]["default"] == 0
    assert "backend" not in schema["properties"]  # Not declared by this callable.
    assert "reward_fn" in schema["x-training-hub"]["unavailable_callable_parameters"]
    assert not {"data_path", "model_path", "reward_fn"} & schema["properties"].keys()


def test_grpo_export_uses_only_the_full_parameter_entrypoint(exported):
    algorithm = next(item for item in exported["algorithms"] if item["name"] == "grpo")
    schema = algorithm["parameter_schema"]
    assert set(schema["properties"]) == {"backend", "num_iterations"}
    assert schema["properties"]["num_iterations"]["default"] == 15
    assert schema["required"] == []


def test_registry_backend_is_not_an_undeclared_public_parameter():
    def grpo(
        model_path: str, data_path: str, ckpt_output_dir: str, num_iterations: int = 15, **kwargs
    ):
        pass

    hub = SimpleNamespace(AlgorithmRegistry=Registry, grpo=grpo)
    exporter = runpy.run_path(str(EXPORTER))
    schema = exporter["describe"](hub, "grpo")["parameter_schema"]
    assert set(schema["properties"]) == {"num_iterations"}
    assert "default_backend" not in schema["x-training-hub"]


def test_closed_public_signature_does_not_advertise_class_only_parameters():
    def future_algorithm(model_path: str, data_path: str, ckpt_output_dir: str, count: int = 0):
        pass

    hub = SimpleNamespace(AlgorithmRegistry=Registry, future_algorithm=future_algorithm)
    exporter = runpy.run_path(str(EXPORTER))
    schema = exporter["describe"](hub, "future_algorithm")["parameter_schema"]
    assert set(schema["properties"]) == {"count"}
    assert schema["required"] == []


@pytest.mark.parametrize(
    "annotation", [tuple[int, str], tuple[float, float], tuple[int], tuple[()]]
)
def test_fixed_length_tuples_are_not_exported_as_unbounded_arrays(annotation):
    exporter = runpy.run_path(str(EXPORTER))
    assert exporter["json_type"](annotation) is None


@pytest.mark.parametrize("annotation", [tuple[int, ...], list[int], Sequence[int]])
def test_homogeneous_sequences_remain_json_arrays(annotation):
    exporter = runpy.run_path(str(EXPORTER))
    assert exporter["json_type"](annotation) == {"type": "array", "items": {"type": "integer"}}


@pytest.mark.parametrize("default", ["preferred", None])
def test_export_does_not_guess_the_backend_default(default):
    class MultipleBackends(Registry):
        @staticmethod
        def list_backends(name):
            return ["verl", "preferred", "other"]

    def entrypoint(backend: str | None = default):
        pass

    hub = SimpleNamespace(AlgorithmRegistry=MultipleBackends, future_algorithm=entrypoint)
    exporter = runpy.run_path(str(EXPORTER))
    schema = exporter["describe"](hub, "future_algorithm")["parameter_schema"]
    if default is None:
        assert "default" not in schema["properties"]["backend"]
        assert "default_backend" not in schema["x-training-hub"]
    else:
        assert schema["properties"]["backend"]["default"] == default
        assert schema["x-training-hub"]["default_backend"] == default


@pytest.mark.parametrize("to_file", [False, True])
def test_exporter_writes_startup_file_or_stdout(tmp_path, exported, monkeypatch, capsys, to_file):
    exporter = runpy.run_path(str(EXPORTER))
    main = exporter["main"]
    monkeypatch.setitem(main.__globals__, "build_catalog", lambda *args: exported)
    monkeypatch.setitem(main.__globals__, "version", lambda name: "unit-version")
    monkeypatch.setitem(sys.modules, "training_hub", SimpleNamespace())
    target = tmp_path / "catalog.json"
    argv = [str(EXPORTER), "unit-runtime"] + ([str(target)] if to_file else [])
    monkeypatch.setattr(sys, "argv", argv)
    main()
    output = capsys.readouterr()
    assert "Discovered 2 algorithms" in output.err
    assert json.loads(target.read_text() if to_file else output.out) == exported
    if to_file:
        assert output.out == ""
        assert not target.with_suffix(".tmp").exists()


def test_failed_export_does_not_replace_existing_catalog(tmp_path, monkeypatch):
    exporter = runpy.run_path(str(EXPORTER))
    main = exporter["main"]

    def failed(*args):
        raise RuntimeError("Training Hub introspection failed")

    monkeypatch.setitem(main.__globals__, "build_catalog", failed)
    monkeypatch.setitem(main.__globals__, "version", lambda name: "unit-version")
    monkeypatch.setitem(sys.modules, "training_hub", SimpleNamespace())
    target = tmp_path / "catalog.json"
    target.write_text("previous catalog")
    monkeypatch.setattr(sys, "argv", [str(EXPORTER), "unit-runtime", str(target)])
    with pytest.raises(RuntimeError, match="introspection failed"):
        main()
    assert target.read_text() == "previous catalog"
