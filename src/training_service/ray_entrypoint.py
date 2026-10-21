"""Code carried in the RayJob entrypoint, executed in the training image.

The submission service embeds this trusted module in an inline Python command.
It is not executed in the API process and receives only non-secret locators.
Credentials are read from project-local Secret volumes inside the Ray pod.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _secret(directory: str, key: str) -> str:
    return (Path(directory) / key).read_text(encoding="utf-8").strip()


def _s3_client(location: dict[str, Any]) -> Any:
    import boto3

    directory = location["credential_dir"]
    kwargs = {
        "aws_access_key_id": _secret(directory, "AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": _secret(directory, "AWS_SECRET_ACCESS_KEY"),
        "endpoint_url": _secret(directory, "AWS_S3_ENDPOINT"),
    }
    region_file = Path(directory) / "AWS_DEFAULT_REGION"
    if region_file.exists():
        kwargs["region_name"] = _secret(directory, "AWS_DEFAULT_REGION")
    session_file = Path(directory) / "AWS_SESSION_TOKEN"
    if session_file.exists():
        kwargs["aws_session_token"] = _secret(directory, "AWS_SESSION_TOKEN")
    return boto3.client("s3", **kwargs)


def _download_s3(location: dict[str, Any], target: Path) -> str:
    client = _s3_client(location)
    bucket = location["bucket"]
    key = location["path"]
    if not key.endswith("/"):
        target.mkdir(parents=True, exist_ok=True)
        file_path = target / Path(key).name
        client.download_file(bucket, key, str(file_path))
        return str(file_path)

    target.mkdir(parents=True, exist_ok=True)
    found = False
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key):
        for item in page.get("Contents", []):
            object_key = item["Key"]
            relative = object_key[len(key) :]
            if not relative or object_key.endswith("/"):
                continue
            parts = relative.split("/")
            if any(part in {"", ".", ".."} or "\\" in part for part in parts):
                raise ValueError("S3 object key has an unsafe relative path")
            destination = target.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, object_key, str(destination))
            found = True
    if not found:
        raise FileNotFoundError("S3 prefix contains no input objects")
    return str(target)


def _input(location: dict[str, Any], kind: str, workdir: Path) -> str:
    scheme = location["scheme"]
    if scheme == "pvc":
        return str(location["mount_path"])
    if scheme == "s3":
        path = Path(_download_s3(location, workdir / kind))
        if kind == "dataset" and path.is_dir():
            candidates = sorted((*path.rglob("*.jsonl"), *path.rglob("*.json")))
            if len(candidates) != 1:
                raise ValueError("S3 dataset prefix must contain one JSON or JSONL file")
            return str(candidates[0])
        return str(path)
    if scheme == "hf":
        token = _secret(location["credential_dir"], location["token_key"])
        if kind == "model":
            from huggingface_hub import snapshot_download

            return str(
                snapshot_download(
                    repo_id=location["repository"],
                    token=token,
                    local_dir=str(workdir / "model"),
                )
            )
        from datasets import load_dataset

        dataset = load_dataset(location["repository"], split="train", token=token)
        target = workdir / "dataset.jsonl"
        dataset.to_json(str(target))
        return str(target)
    raise ValueError("Unsupported input source")


def _dataset_format(path: str) -> str | None:
    """Recognize the contract example's Alpaca JSONL without a public format knob."""
    if not path.endswith(".jsonl"):
        return None
    with Path(path).open(encoding="utf-8") as source:
        first_line = next((line for line in source if line.strip()), "")
    if not first_line:
        return None
    first_row = json.loads(first_line)
    if isinstance(first_row, dict) and {"instruction", "output"} <= first_row.keys():
        return "alpaca"
    return None


def _upload_s3(location: dict[str, Any], directory: Path) -> None:
    client = _s3_client(location)
    prefix = location["path"].rstrip("/") + "/"
    for path in directory.rglob("*"):
        if path.is_file():
            client.upload_file(
                str(path),
                location["bucket"],
                prefix + path.relative_to(directory).as_posix(),
            )


def main(configuration: dict[str, Any]) -> None:
    """Resolve input locators, run Training Hub LoRA, and save optional output."""
    from training_hub import lora_sft

    workdir = Path("/tmp/training-api")
    workdir.mkdir(parents=True, exist_ok=True)
    model_path = _input(configuration["model"], "model", workdir)
    data_path = _input(configuration["dataset"], "dataset", workdir)
    output = configuration.get("output")
    output_dir = (
        Path(output["mount_path"])
        if output is not None and output["scheme"] == "pvc"
        else workdir / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = dict(configuration["parameters"])
    dataset_format = _dataset_format(data_path)
    if dataset_format is not None:
        parameters["dataset_type"] = dataset_format
    lora_sft(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=str(output_dir),
        **parameters,
    )
    if output is not None and output["scheme"] == "s3":
        _upload_s3(output, output_dir)
