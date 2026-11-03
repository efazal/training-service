"""Shared preflight checks for the live suite and deployment runner."""

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

PROFILES = ("no-kueue", "with-kueue", "unqueued-opt-out")


def namespace(value):
    if len(value) > 63 or not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", value):
        raise ValueError("Project must be a Kubernetes namespace name")
    return value


def workload(filename, profile):
    try:
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("Job file must contain a readable JSON request object") from None
    if not isinstance(payload, dict):
        raise ValueError("Job file must contain a JSON object")
    if profile not in PROFILES:
        raise ValueError("Training requires an explicit --queue-profile")
    queue = payload.get("queue_name")
    if profile == "with-kueue":
        if not isinstance(queue, str) or not queue:
            raise ValueError("with-kueue requires a non-empty queue_name in the job file")
        namespace(queue)
    elif "queue_name" in payload:
        raise ValueError("Unqueued profiles must omit queue_name, not supply an empty string")
    if payload.get("backend", "ray") != "ray":
        raise ValueError("This live lifecycle currently verifies the Ray backend only")
    for field in ("model", "dataset", "algorithm", "output"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"Training fixture requires {field}")
    for field, key in (("model", "uri"), ("dataset", "uri"), ("output", "destination_uri")):
        value = payload[field].get(key)
        if not isinstance(value, str):
            raise ValueError(f"Training fixture requires {field}.{key}")
        parsed = urlsplit(value)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "Fixtures must use project-local references, never embedded credentials"
            )
    return payload


def pvc_path(uri):
    parsed = urlsplit(uri)
    path = unquote(parsed.path).lstrip("/")
    if (
        parsed.scheme != "pvc"
        or not parsed.netloc
        or parsed.username
        or parsed.query
        or parsed.fragment
        or not path
        or parsed.path.startswith("//")
        or any(part in {"", ".", ".."} for part in path.rstrip("/").split("/"))
        or "\\" in path
        or any(ord(c) < 32 or ord(c) == 127 for c in path)
    ):
        raise ValueError("Artifact verification requires a safe pvc://claim/directory output")
    namespace(parsed.netloc)
    return parsed.netloc, path
