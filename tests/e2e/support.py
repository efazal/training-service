"""Shared preflight checks for the live suite and deployment runner."""

import json
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote, urlsplit

PROFILES = ("no-kueue", "with-kueue", "default-queue", "unqueued-opt-out")
REMOTE_SOURCES = ("s3", "hf")


def api_token():
    """Use an explicit caller token, otherwise capture the current oc token in memory."""
    if token := os.getenv("TRAINING_SERVICE_TOKEN", "").strip():
        return token
    try:
        result = subprocess.run(
            ["oc", "whoami", "-t"],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=15,
        )
        token = result.stdout.strip()
        if result.returncode == 0 and token and not any(char.isspace() for char in token):
            return token
    except (OSError, subprocess.SubprocessError):
        pass
    raise ValueError(
        "Cannot obtain an API token from the current oc session. Run oc login for the "
        "target cluster, or set TRAINING_SERVICE_TOKEN to an explicit API caller token. "
        "Raw credential output withheld."
    )


def selector_matches(labels, selector):
    if any(labels.get(k) != v for k, v in selector.get("matchLabels", {}).items()):
        return False
    for item in selector.get("matchExpressions", []):
        key, operator, values = item["key"], item["operator"], item.get("values", [])
        if operator == "In" and labels.get(key) not in values:
            return False
        if operator == "NotIn" and labels.get(key) in values:
            return False
        if operator == "Exists" and key not in labels:
            return False
        if operator == "DoesNotExist" and key in labels:
            return False
        if operator not in {"In", "NotIn", "Exists", "DoesNotExist"}:
            return False
    return True


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
        raise ValueError(
            "Unqueued and default-queue profiles must omit queue_name entirely "
            "(not null, empty or a name)"
        )
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


def s3_path(uri):
    """Return the bucket and key from a connection-backed, credential-free S3 URI."""
    parsed = urlsplit(uri)
    path = unquote(parsed.path).lstrip("/")
    bucket, separator, key = path.partition("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.startswith("//")
        or not separator
        or not bucket
        or not key
        or any(part in {"", ".", ".."} for part in key.rstrip("/").split("/"))
        or "\\" in key
        or any(ord(char) < 32 or ord(char) == 127 for char in key)
    ):
        raise ValueError("Remote E2E output requires a safe s3://connection/bucket/prefix URI")
    return bucket, key


def remote_workloads(filename):
    """Build the four remote-source combinations from one credential-free fixture."""
    try:
        fixture = json.loads(Path(filename).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("Remote fixture file must contain readable JSON") from None
    if not isinstance(fixture, dict):
        raise ValueError("Remote fixture file must contain a JSON object")
    base = fixture.get("base_request")
    if not isinstance(base, dict) or any(field in base for field in ("name", "model", "dataset")):
        raise ValueError("base_request must omit name, model and dataset")
    if base.get("backend", "ray") != "ray" or not isinstance(base.get("algorithm"), dict):
        raise ValueError("Remote E2E requires a Ray algorithm in base_request")
    if base["algorithm"].get("name") != "lora_sft":
        raise ValueError("Remote E2E verifies LoRA adapter output; use lora_sft")
    output = base.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("destination_uri"), str):
        raise ValueError("Remote E2E requires output.destination_uri")
    s3_path(output["destination_uri"])
    if not output["destination_uri"].endswith("/"):
        raise ValueError("Remote E2E S3 output must be a directory prefix ending in /")
    for field in ("models", "datasets"):
        choices = fixture.get(field)
        if not isinstance(choices, dict) or set(choices) != set(REMOTE_SOURCES):
            raise ValueError(f"{field} must define exactly s3 and hf")
        for source, spec in choices.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("uri"), str):
                raise ValueError(f"{field}.{source} requires a uri string")
            parsed = urlsplit(spec["uri"])
            if (
                parsed.scheme != source
                or not parsed.netloc
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"{field}.{source} must use a credential-free {source} URI")
            if source == "s3":
                s3_path(spec["uri"])
                if field == "datasets" and ("config" in spec or "split" in spec):
                    raise ValueError("S3 datasets cannot use Hugging Face config or split")
            elif len(parsed.path.strip("/").split("/")) != 2:
                raise ValueError(f"{field}.hf must identify an owner/repository")
    result = {}
    for model_source in REMOTE_SOURCES:
        for dataset_source in REMOTE_SOURCES:
            payload = deepcopy(base)
            payload["model"] = deepcopy(fixture["models"][model_source])
            payload["dataset"] = deepcopy(fixture["datasets"][dataset_source])
            result[f"{model_source}_model_{dataset_source}_dataset"] = payload
    return result
