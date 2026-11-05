"""No Training Hub, GPU or cluster needed to verify registry introspection."""

import json
import runpy
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.e2e.environment import RunEnvironment

from training_service.catalog import validate_catalog

EXPORTER = Path(__file__).resolve().parents[2] / "scripts/export_algorithm_catalog.py"


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
    assert schema["properties"]["backend"]["enum"] == ["engine"]
    assert "reward_fn" in schema["x-training-hub"]["unavailable_callable_parameters"]
    assert not {"data_path", "model_path", "reward_fn"} & schema["properties"].keys()


def test_grpo_export_uses_only_the_full_parameter_entrypoint(exported):
    algorithm = next(item for item in exported["algorithms"] if item["name"] == "grpo")
    schema = algorithm["parameter_schema"]
    assert set(schema["properties"]) == {"backend", "num_iterations"}
    assert schema["properties"]["num_iterations"]["default"] == 15
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


def test_e2e_catalog_is_created_in_owned_namespace(tmp_path, exported, monkeypatch):
    monkeypatch.setenv("TRAINING_RAY_IMAGE", "unrelated-shell-image")
    args = SimpleNamespace(
        namespace="test-project", runtime_image="unit-runtime", use_podman_auth=False
    )
    command = Mock(return_value=json.dumps(exported))
    environment = RunEnvironment(args, "run-123", tmp_path, command)
    environment.create = Mock()
    environment.provision_catalog()
    argv = command.call_args.args[0]
    assert argv[:5] == ["oc", "exec", "-n", "test-project", "e2e-output-verifier"]
    document = environment.create.call_args.args[1]
    assert document["kind"] == "ConfigMap"
    assert document["metadata"]["name"] == "e2e-algorithms"
    assert document["metadata"]["labels"]["training.rhoai.io/e2e-run"] == "run-123"
    assert json.loads(document["data"]["catalog.json"]) == exported


def test_e2e_rejects_export_from_a_different_image(tmp_path, exported):
    args = SimpleNamespace(
        namespace="test-project", runtime_image="different-image", use_podman_auth=False
    )
    environment = RunEnvironment(args, "run-123", tmp_path, Mock(return_value=json.dumps(exported)))
    environment.create = Mock()
    with pytest.raises(RuntimeError, match="valid catalog"):
        environment.provision_catalog()
    environment.create.assert_not_called()


def test_e2e_rejects_bad_runtime_export(tmp_path):
    args = SimpleNamespace(
        namespace="test-project", runtime_image="unit-runtime", use_podman_auth=False
    )
    environment = RunEnvironment(args, "run-123", tmp_path, Mock(return_value="not-json"))
    environment.create = Mock()
    with pytest.raises(RuntimeError, match="valid catalog"):
        environment.provision_catalog()
    environment.create.assert_not_called()
