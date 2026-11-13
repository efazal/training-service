# Training image and algorithm requirements

Use this optional reference when changing the training image or preparing a
training request. For installation and the API URL, start with the
[installation guide](installation.md).

## What is the training runtime?

The runtime is the container image that executes training jobs. It is separate
from the image running the REST API:

| Chart setting | Purpose |
| --- | --- |
| `image.repository` and `image.tag` | Run Training Service and expose the API. |
| `trainingRuntime.image` | Run model training with Ray, Training Hub, and GPU libraries. |

Algorithm discovery and submission validation share a catalog generated
automatically from this runtime image, without a GPU or manual catalog setup.

## Choose a compatible image

- The default runtime is `quay.io/modh/ray:2.58.0-py312-cu130-th0112`.
- The API pins Ray 2.58.0 and CodeFlare SDK 0.40.1. Use a compatible Ray version,
  Training Hub dependencies, GPU drivers, and node architecture. Check the actual
  versions inside the image, as tags can differ from installed versions.
- Prefer a tested image digest when customizing `trainingRuntime.image`. Changing
  it regenerates the catalog when the API pods restart. See
  [algorithm discovery](algorithm-discovery.md) for setup troubleshooting.

**GRPO compatibility:** the tested Training Hub 0.11.2 image failed during GRPO
training because of a TRL/verl import incompatibility. Verify a compatible
runtime before using GRPO; successful submission does not guarantee completion.

## Supported algorithms and resources

| Algorithm | Supported layout | Main requirement |
| --- | --- | --- |
| `sft` | One training node: `num_nodes=1`. | Multiple GPUs use local torchrun processes. |
| `osft` | One training node: `num_nodes=1`. | Supply all required OSFT parameters below. |
| `lora_sft` | One training node: `num_nodes=1`. | Multiple GPUs require `enable_model_splitting=true`. |
| `lora_grpo` | GPU-free head and at least one GPU worker: `num_nodes>=2`. | Shared ReadWriteMany PVC storage and a compatible verl runtime. |

For SFT, OSFT, and LoRA SFT, `gpus_per_node` is the GPU count on the training
node. Defaults are 1 GPU, 4 CPUs, and 16Gi memory; size them for your model.
For GRPO, `num_nodes` includes the head and `gpus_per_node` applies to each worker.
CPU/memory settings size the workers; the head uses 4 CPUs and 32Gi.

### Algorithm-specific settings

- **OSFT:** provide `unfreeze_rank_ratio`, `effective_batch_size`,
  `max_tokens_per_gpu`, `max_seq_len`, and `learning_rate`.
- **LoRA SFT:** bf16 is the default and needs an Ampere or newer GPU. For older
  GPUs such as T4, set `bf16: false` and `fp16: true`. Check the runtime's
  `assistant_only_loss` default against your dataset format.
- **LoRA GRPO:** use `prompt_batch_size`; `tasks_per_iteration` is not supported.
  GPU counts and batch/parallelism settings must satisfy submission validation.

Use `GET /api/v1/algorithms/{name}` for the selected image's required parameters,
types, and defaults. Invalid parameters or layouts return `422`. Additional
algorithms can appear in discovery but cannot yet be submitted through this API.

## Prepare datasets and persistent storage

Prepare model and dataset files before submission. Every new job requires an
S3 or PVC output destination; the API references existing project storage.

- **SFT, OSFT, and LoRA SFT:** Hugging Face `dataset.config` selects a named dataset
  configuration; `dataset.split` selects a split or slice and defaults to `train`.
- **LoRA GRPO:** `dataset.config` and `dataset.split` are rejected. Where the runtime
  supports it, `algorithm.parameters.data_config` selects the HF configuration for
  tool-call datasets. It is not a column mapping or an option for local math data.
- **GRPO storage:** use shared ReadWriteMany PVC output and shared PVC or HF inputs.
  S3 inputs/output are unsupported for this layout. HF inputs must reference the
  same existing project Secret, including for public repositories. Math rewards
  require a PVC JSONL dataset.

See the [API integration quickstart](dashboard-integration.md) for a submission
example and the [Ray fine-tuning examples](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/fine-tuning)
for dataset preparation.
