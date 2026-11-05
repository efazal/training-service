"""Metadata inspection and dispatch never require a real training installation."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training_service import ray_entrypoint


@pytest.mark.parametrize("name", ["sft", "osft", "lora_sft", "grpo", "lora_grpo"])
def test_entrypoint_calls_the_selected_public_algorithm(monkeypatch, tmp_path, name):
    calls = []

    def train(model_path, data_path, ckpt_output_dir, warmup_steps=10):
        calls.append((model_path, data_path, ckpt_output_dir, warmup_steps))

    hub = SimpleNamespace(AlgorithmRegistry=SimpleNamespace(list_algorithms=lambda: [name]))
    setattr(hub, name, train)
    monkeypatch.setitem(sys.modules, "training_hub", hub)
    monkeypatch.setattr(
        ray_entrypoint,
        "Path",
        lambda value: tmp_path if value == "/tmp/training-api" else Path(value),
    )
    monkeypatch.setattr(ray_entrypoint, "_input", lambda location, kind, workdir: kind + "-path")
    monkeypatch.setattr(ray_entrypoint, "_dataset_format", lambda path: None)
    ray_entrypoint.main(
        {"algorithm": name, "model": {}, "dataset": {}, "parameters": {"warmup_steps": 0}}
    )
    assert calls == [("model-path", "dataset-path", str(tmp_path / "output"), 0)]


def test_changed_runtime_is_rejected_before_input_download(monkeypatch):
    hub = SimpleNamespace(AlgorithmRegistry=SimpleNamespace(list_algorithms=lambda: []))
    monkeypatch.setitem(sys.modules, "training_hub", hub)
    inputs = Mock()
    monkeypatch.setattr(ray_entrypoint, "_input", inputs)
    with pytest.raises(ValueError, match="no longer available"):
        ray_entrypoint.main({"algorithm": "grpo"})
    inputs.assert_not_called()
