"""A tiny exported runtime catalog for isolated API/backend tests."""

import json

import pytest

from training_service import catalog


@pytest.fixture(autouse=True)
def runtime_catalog(tmp_path_factory, monkeypatch):
    payload = {
        "runtime": {
            "image": "quay.io/example/ray:2.55.1",
            "training_hub_version": "0.8.1",
            "registry_sha256": "a" * 64,
        },
        "algorithms": [
            {
                "name": "lora_sft",
                "supported_backends": ["ray"],
                "parameter_schema": {
                    "type": "object",
                    "properties": {
                        "learning_rate": {"type": "number"},
                        "num_epochs": {"type": "integer"},
                        "effective_batch_size": {"type": "integer"},
                        "max_seq_len": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
            }
        ],
    }
    # Keep unrelated tests' tmp_path empty (TLS/credential tests verify that).
    path = tmp_path_factory.mktemp("runtime-catalog") / "catalog.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("TRAINING_SERVICE_CATALOG_PATH", str(path))
    monkeypatch.delenv("TRAINING_RAY_IMAGE", raising=False)
    catalog._load_catalog.cache_clear()
    yield path
    catalog._load_catalog.cache_clear()
