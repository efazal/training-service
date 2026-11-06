"""Code carried in the RayJob entrypoint, executed in the training image.

The submission service embeds this trusted module in an inline Python command.
It is not executed in the API process and receives only non-secret locators.
Credentials are read from project-local Secret volumes inside the Ray pod.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from importlib import invalidate_caches
from pathlib import Path
from typing import Any

SUPPORTED_DATASET_EXTENSIONS = frozenset({".jsonl", ".json", ".csv", ".parquet"})


class TrainingBackend:
    """One supported Training Hub execution profile, shared by API and Ray driver."""

    name = ""
    backend_name = ""
    remote_hf = False

    def required_dataset_columns(self, parameters: dict[str, Any], columns: set[str]) -> set[str]:
        """Return columns that can be checked without reading training rows."""
        return set()

    def validate_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        return values

    def validate_resources(
        self, gpu_count: int, num_nodes: int, gpu_type: str, parameters: dict[str, Any]
    ) -> None:
        if gpu_count == 0:
            raise ValueError("The selected Training Hub algorithm requires GPUs.")

    def validate_locations(
        self, locations: dict[str, dict[str, str]], parameters: dict[str, Any]
    ) -> None:
        pass

    def cluster_options(
        self,
        gpu_count: int,
        cpu: str,
        memory: str,
        resolved: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "head_gpu_count": gpu_count,
            "head_cpu": cpu,
            "head_memory": memory,
            "envs": {},
        }

    def training_options(
        self,
        data_path: str,
        output_dir: Path,
        parameters: dict[str, Any],
        gpus_per_node: int,
        num_workers: int,
    ) -> dict[str, Any]:
        return dict(parameters)

    def model_path(self, path: str) -> str:
        """Return the model location expected by this Training Hub backend."""
        return path

    def run(
        self,
        model_path: str,
        data_path: str,
        output_dir: Path,
        parameters: dict[str, Any],
        gpus_per_node: int,
        num_workers: int,
    ) -> Any:
        import training_hub

        algorithm = training_hub.create_algorithm(self.name, self.backend_name)
        return algorithm.train(
            model_path=self.model_path(model_path),
            data_path=data_path,
            ckpt_output_dir=str(output_dir),
            **self.training_options(data_path, output_dir, parameters, gpus_per_node, num_workers),
        )


class SingleNodeTrainingBackend(TrainingBackend):
    """Head-local launchers must not silently allocate unused Ray workers."""

    def validate_resources(
        self, gpu_count: int, num_nodes: int, gpu_type: str, parameters: dict[str, Any]
    ) -> None:
        super().validate_resources(gpu_count, num_nodes, gpu_type, parameters)
        if num_nodes != 1:
            raise ValueError(f"{self.name} currently requires resources.num_nodes=1.")


class SFTBackend(SingleNodeTrainingBackend):
    name = "sft"
    backend_name = "instructlab-training"

    def required_dataset_columns(self, parameters: dict[str, Any], columns: set[str]) -> set[str]:
        if parameters.get("is_pretraining"):
            return {parameters.get("document_column_name") or "document"}
        return {"messages"}

    def validate_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        _validate_pretraining(values)
        if ("beta1" in values) != ("beta2" in values):
            raise ValueError("SFT beta1 and beta2 must be set together.")
        return values

    def training_options(
        self,
        data_path: str,
        output_dir: Path,
        parameters: dict[str, Any],
        gpus_per_node: int,
        num_workers: int,
    ) -> dict[str, Any]:
        options = dict(parameters)
        data_output_dir = output_dir / "data"
        data_output_dir.mkdir(parents=True, exist_ok=True)
        options["data_output_dir"] = str(data_output_dir)
        options["nproc_per_node"] = gpus_per_node
        options["nnodes"] = 1
        return options


class OSFTBackend(SFTBackend):
    name = "osft"
    backend_name = "mini-trainer"

    def required_dataset_columns(self, parameters: dict[str, Any], columns: set[str]) -> set[str]:
        if parameters.get("use_processed_dataset"):
            return set()
        return super().required_dataset_columns(parameters, columns)

    def validate_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        _validate_pretraining(values)
        if values.get("is_pretraining") and values.get("unmask_messages"):
            raise ValueError("OSFT pretraining cannot unmask messages.")
        # Both supported Training Hub versions strictly type-check float parameters.
        for key in (
            "unfreeze_rank_ratio",
            "learning_rate",
            "beta1",
            "beta2",
            "eps",
            "weight_decay",
        ):
            if values.get(key) is not None:
                try:
                    converted = float(values[key])
                except OverflowError:
                    raise ValueError(f"OSFT {key} must be a finite number.") from None
                if not math.isfinite(converted):
                    raise ValueError(f"OSFT {key} must be a finite number.")
                values[key] = converted
        return values


class LoRASFTBackend(SingleNodeTrainingBackend):
    name = "lora_sft"
    backend_name = "unsloth"

    def validate_parameters(self, values: dict[str, Any]) -> dict[str, Any]:
        if values.get("load_in_4bit") and values.get("load_in_8bit"):
            raise ValueError("Choose only one quantization mode.")
        if values.get("bf16", True) and values.get("fp16", False):
            raise ValueError("fp16 requires bf16=false; choose only one precision mode.")
        return values

    def validate_resources(
        self, gpu_count: int, num_nodes: int, gpu_type: str, parameters: dict[str, Any]
    ) -> None:
        super().validate_resources(gpu_count, num_nodes, gpu_type, parameters)
        if gpu_count > 1 and not parameters.get("enable_model_splitting"):
            raise ValueError("Multi-GPU lora_sft requires enable_model_splitting=true.")

    def training_options(
        self,
        data_path: str,
        output_dir: Path,
        parameters: dict[str, Any],
        gpus_per_node: int,
        num_workers: int,
    ) -> dict[str, Any]:
        options = dict(parameters)
        if "dataset_type" not in options:
            dataset_format = _dataset_format(data_path)
            if dataset_format is not None:
                options["dataset_type"] = dataset_format
        return options


class LoRAGRPOBackend(TrainingBackend):
    name = "lora_grpo"
    backend_name = "verl"
    remote_hf = True

    def model_path(self, path: str) -> str:
        # verl treats a trailing slash on local model directories as an invalid
        # model identifier; other Training Hub backends accept that path.
        return str(Path(path))

    def validate_resources(
        self, gpu_count: int, num_nodes: int, gpu_type: str, parameters: dict[str, Any]
    ) -> None:
        super().validate_resources(gpu_count, num_nodes, gpu_type, parameters)
        if num_nodes < 2:
            raise ValueError("lora_grpo requires a GPU-free Ray head and at least one GPU worker.")
        if gpu_type != "nvidia.com/gpu":
            raise ValueError("lora_grpo with verl requires NVIDIA GPUs.")
        parallelism = parameters.get("tensor_parallel_size", 1)
        if parallelism < 1 or parallelism > gpu_count or gpu_count % parallelism:
            raise ValueError("tensor_parallel_size must be positive and divide gpus_per_node.")
        group_size = parameters.get("group_size", 8)
        prompt_batch_size = parameters.get("prompt_batch_size", 100)
        total_gpus = gpu_count * (num_nodes - 1)
        if (prompt_batch_size * group_size) % (total_gpus * 4):
            raise ValueError(
                "prompt_batch_size * group_size must be divisible by 4 * "
                "gpus_per_node * (num_nodes - 1)."
            )
        # Training Hub derives the PPO minibatch from the prompt batch and
        # rounds it up to at least one GPU's worth when it is too small.
        ppo_minibatch = min(prompt_batch_size, 256)
        if ppo_minibatch % gpu_count:
            ppo_minibatch = max(gpu_count, (ppo_minibatch // gpu_count) * gpu_count)
        if prompt_batch_size < ppo_minibatch:
            raise ValueError(
                "prompt_batch_size must be at least the derived PPO minibatch size "
                f"({ppo_minibatch})."
            )

    def validate_locations(
        self, locations: dict[str, dict[str, str]], parameters: dict[str, Any]
    ) -> None:
        if "output" not in locations or locations["output"]["scheme"] != "pvc":
            raise ValueError("lora_grpo requires a shared PVC output destination.")
        if any(locations[role]["scheme"] == "s3" for role in ("model", "dataset")):
            raise ValueError("lora_grpo inputs must be shared PVCs or Hugging Face repositories.")
        dataset = locations["dataset"]
        if parameters.get("reward_type") == "math" and dataset["scheme"] == "hf":
            raise ValueError("lora_grpo with reward_type=math requires a PVC JSONL dataset.")
        if (
            parameters.get("reward_type") == "math"
            and dataset["scheme"] == "pvc"
            and not dataset["path"].endswith(".jsonl")
        ):
            raise ValueError("lora_grpo with reward_type=math requires a PVC JSONL dataset.")
        if dataset["scheme"] == "pvc" and not dataset["path"].endswith((".json", ".jsonl")):
            raise ValueError("lora_grpo PVC dataset must be a JSON or JSONL file.")
        hf_secrets = {
            locations[role]["name"]
            for role in ("model", "dataset")
            if locations[role]["scheme"] == "hf"
        }
        if len(hf_secrets) > 1:
            raise ValueError("lora_grpo Hugging Face inputs must share one credential Secret.")

    def cluster_options(
        self, gpu_count: int, cpu: str, memory: str, resolved: dict[str, dict[str, str]]
    ) -> dict[str, Any]:
        hf_paths = {
            location["credential_dir"] + "/" + location["token_key"]
            for location in resolved.values()
            if location["scheme"] == "hf"
        }
        return {
            "head_gpu_count": 0,
            "head_cpu": "4",
            "head_memory": "32Gi",
            "envs": {"HF_TOKEN_PATH": next(iter(hf_paths))} if hf_paths else {},
        }

    def training_options(
        self,
        data_path: str,
        output_dir: Path,
        parameters: dict[str, Any],
        gpus_per_node: int,
        num_workers: int,
    ) -> dict[str, Any]:
        return {**parameters, "n_gpus": gpus_per_node, "nnodes": num_workers}


def _validate_pretraining(values: dict[str, Any]) -> None:
    if values.get("is_pretraining") and "block_size" not in values:
        raise ValueError("block_size is required for pretraining.")
    if "block_size" in values and not values.get("is_pretraining"):
        raise ValueError("block_size requires pretraining mode.")


_BACKENDS = {
    backend.name: backend
    for backend in (SFTBackend(), OSFTBackend(), LoRASFTBackend(), LoRAGRPOBackend())
}


def get_backend(name: str) -> TrainingBackend:
    """Reject anything not explicitly supported by this submission adapter."""
    try:
        return _BACKENDS[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Training Hub algorithm '{name}': no execution profile."
        ) from error


def _secret(directory: str, key: str) -> str:
    return (Path(directory) / key).read_text(encoding="utf-8").strip()


def _s3_modules_available() -> bool:
    try:
        import boto3  # noqa: F401
        import botocore  # noqa: F401
        from boto3.s3 import transfer  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_s3_runtime(workdir: Path) -> None:
    """Install the pinned S3 client only when the selected image lacks it."""
    if _s3_modules_available():
        return
    packages = workdir / "site-packages"
    try:
        packages.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            "S3 support is unavailable: the job-local package directory is not writable."
        ) from error
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--target",
                str(packages),
                "boto3==1.40.61",
                "botocore==1.40.61",
            ],
            check=False,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            "S3 support is unavailable: the runtime image lacks boto3 and its "
            "job-local installation failed. Check pip and package index or mirror access."
        ) from error
    if result.returncode:
        raise RuntimeError(
            "S3 support is unavailable: the runtime image lacks boto3 and its "
            "job-local installation failed. Check pip and package index or mirror access."
        )
    sys.path.insert(0, str(packages))
    invalidate_caches()
    if not _s3_modules_available():
        raise RuntimeError("S3 support is unavailable after dependency installation.")


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


def _input(location: dict[str, Any], kind: str, workdir: Path, *, remote_hf: bool = False) -> str:
    scheme = location["scheme"]
    if scheme == "pvc":
        return str(location["mount_path"])
    if scheme == "s3":
        path = Path(_download_s3(location, workdir / kind))
        if kind == "dataset" and path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_DATASET_EXTENSIONS
            )
            if len(candidates) != 1:
                raise ValueError("S3 dataset prefix must contain exactly one supported data file")
            return str(candidates[0])
        return str(path)
    if scheme == "hf":
        if remote_hf:
            # verl loads the model on Ray workers and streams the dataset. A
            # head-local snapshot/JSONL would not be accessible from workers.
            return str(location["repository"])
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

        dataset = load_dataset(
            location["repository"],
            name=location.get("config"),
            split=location.get("split", "train"),
            token=token,
        )
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


def _run_training(
    algorithm: str,
    model_path: str,
    data_path: str,
    output_dir: Path,
    parameters: dict[str, Any],
    gpus_per_node: int,
    num_workers: int,
) -> Any:
    """Dispatch to the selected backend, which owns its entrypoint options."""
    return get_backend(algorithm).run(
        model_path, data_path, output_dir, parameters, gpus_per_node, num_workers
    )


def main(configuration: dict[str, Any]) -> None:
    """Resolve input locators, execute the selected algorithm, and save output."""

    workdir = Path("/tmp/training-api")
    workdir.mkdir(parents=True, exist_ok=True)
    if any(
        configuration.get(role, {}).get("scheme") == "s3" for role in ("model", "dataset", "output")
    ):
        _ensure_s3_runtime(workdir)
    algorithm = configuration["algorithm"]
    remote_hf = get_backend(algorithm).remote_hf
    model_path = _input(configuration["model"], "model", workdir, remote_hf=remote_hf)
    data_path = _input(configuration["dataset"], "dataset", workdir, remote_hf=remote_hf)
    output = configuration.get("output")
    output_dir = (
        Path(output["mount_path"])
        if output is not None and output["scheme"] == "pvc"
        else workdir / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_training(
        algorithm,
        model_path,
        data_path,
        output_dir,
        dict(configuration["parameters"]),
        configuration["gpus_per_node"],
        configuration["num_workers"],
    )
    if output is not None and output["scheme"] == "s3":
        _upload_s3(output, output_dir)
