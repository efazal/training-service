"""Runs in a test-owned init container; downloads only a pinned public model."""

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

os.umask(0o002)
try:
    with tempfile.TemporaryFile(dir="/data"):
        pass
except OSError:
    raise RuntimeError(
        "E2E fixture PVC is not writable by the non-root container; "
        "check the namespace-allocated fsGroup and NFS volume permissions"
    ) from None
print("Fixture PVC write check passed; downloading the pinned public model", flush=True)

snapshot_download(
    repo_id=MODEL,
    revision=REVISION,
    token=False,
    local_dir="/data/models/qwen",
    ignore_patterns=["*.bin", "*.gguf", "*.onnx", "*.h5", "*.msgpack"],
)
dataset = Path("/data/datasets/tiny-chat.jsonl")
dataset.parent.mkdir(parents=True, exist_ok=True)
rows = [
    {
        "messages": [
            {"role": "user", "content": f"What is {i} plus one?"},
            {"role": "assistant", "content": f"{i} plus one is {i + 1}."},
        ]
    }
    for i in range(8)
]
dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
Path("/data/outputs").mkdir(exist_ok=True)
print("Pinned public model and eight training records prepared.", flush=True)
