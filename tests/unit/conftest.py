"""A real exported catalog for isolated API/backend tests, never a production fallback.

The snapshot comes from the retained Training Hub 0.8.1 discovery deployment;
docstrings are omitted for size. Only the fake SDK's image identity is replaced.
"""

import json
from pathlib import Path

import pytest

from training_service import catalog


@pytest.fixture(autouse=True)
def runtime_catalog(tmp_path_factory, monkeypatch):
    source = Path(__file__).parent / "fixtures/training-hub-0.8.1-catalog.json"
    payload = json.loads(source.read_text())
    payload["runtime"]["image"] = "quay.io/example/ray:2.58.0"
    # Keep unrelated tests' tmp_path empty (TLS/credential tests verify that).
    path = tmp_path_factory.mktemp("runtime-catalog") / "catalog.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("TRAINING_SERVICE_CATALOG_PATH", str(path))
    monkeypatch.delenv("TRAINING_RAY_IMAGE", raising=False)
    catalog._load_catalog.cache_clear()
    yield path
    catalog._load_catalog.cache_clear()
