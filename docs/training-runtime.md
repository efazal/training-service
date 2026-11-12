# Training runtime and compatibility

The chart automatically discovers Training Hub algorithms and parameters from
the configured runtime image, using a GPU-free init container. No manual catalog
export or ConfigMap setup is needed. GET discovery and POST validation share
`catalog.json`; unsupported algorithms or parameters return `422` before submission.

POST dispatches through per-algorithm adapters to Training Hub algorithm/backend classes
used by the [Ray fine-tuning examples](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/fine-tuning):
`sft`, `osft`, `lora_sft`, and `lora_grpo`. Algorithm overrides are validated
against the generated runtime catalog, not a static service schema or OpenAPI. Unsupported or
malformed overrides return `422` rather than being passed to Training Hub.
Each adapter owns its additional validation, data preparation, and Ray
entrypoint settings. Training Hub's `create_algorithm` selects the explicit backend
class; no single convenience function is hardcoded. New Training Hub algorithms
are not exposed automatically.

OSFT requires `unfreeze_rank_ratio`, `effective_batch_size`,
`max_tokens_per_gpu`, `max_seq_len`, and `learning_rate`. SFT and OSFT derive
`data_output_dir` beneath the output directory. Model/data preparation shown
in the notebooks is still performed before calling the API; the request uses
existing project storage or Hugging Face repositories.
For Hugging Face datasets loaded on the Ray head, `dataset.config` selects
a named dataset configuration and `dataset.split` selects a split (default:
`train`). These options are not used by the remote GRPO loader.

For SFT, OSFT, and LoRA SFT, the Ray head runs training and receives
`resources.gpus_per_node` GPUs, matching the one-node examples. For LoRA GRPO,
`resources.num_nodes` includes a GPU-free head and at least one GPU worker;
`gpus_per_node` is the GPU count **per worker**. The adapter derives Training
Hub's `n_gpus` and `nnodes` from these resources and selects the `verl`
backend. GRPO's head uses 4 CPUs/32Gi; CPU and memory fields size each worker.
SFT, OSFT, and LoRA SFT require `num_nodes=1`; unsupported multi-node
requests return `422` before resource lookup or submission. SFT/OSFT derive
local torchrun process counts from `gpus_per_node`. Multi-GPU LoRA requires
explicit `enable_model_splitting=true`; no distributed workers are launched.
LoRA SFT defaults to bf16, which needs an Ampere or newer GPU. On older GPUs
such as the T4 (AWS g4dn), set `bf16: false` and `fp16: true`; otherwise the job
fails with `Your setup doesn't support bf16/gpu`.
GRPO requires a shared ReadWriteMany PVC output and either shared PVC or HF
inputs; S3 inputs/output are not supported for this topology. HF inputs must
reference an existing project Secret, even for public repos. GRPO uses one
such Secret for both HF inputs and mounts its token file on all Ray pods
without serializing the token into the RayJob. The GRPO notebook's
`tasks_per_iteration` argument is not consumed by the reviewed Training Hub
versions and is therefore rejected by this API; use `prompt_batch_size`
instead. GRPO math rewards require a PVC JSONL dataset; Training Hub cannot
open a Hugging Face repository ID as a local math-data file. The exposed
parameters come from the selected runtime's signatures. Text LoRA's
`assistant_only_loss` default differs between those runtimes; test the chosen
image with the intended dataset format.

The API maps resources and queue settings to a CodeFlare RayJob.
When resources are omitted, the default per-node CPU and memory limits are
4 CPUs and 16Gi for the training node; the GRPO head keeps its dedicated
4-CPU/32Gi layout. Callers should size worker resources for their model and
algorithm.
The `202` response returns the RayJob name as `id` and a project-scoped
`Location` header; it does not mean training has completed.

The chart's dedicated `trainingRuntime.image` value selects the Ray workload
image, separate from `image.repository`/`image.tag` for the API service. For
non-Helm deployments, set the process environment variable `TRAINING_RAY_IMAGE`.
The image must contain Ray and Training Hub with a Ray
version compatible with the API's Ray 2.58.0. The API image pins the RHOAI 3.6
CodeFlare SDK 0.40.1 and Ray 2.58.0 wheels. Discovery, readiness and submission
return `503` without a valid startup catalog. CodeFlare accepts
a caller-scoped Kubernetes client; no service-side job store or hand-authored
RayJob manifest is used.

The four Training Hub algorithms in the referenced Ray fine-tuning examples
are `sft`, `osft`, `lora_sft`, and `lora_grpo`. Discovery can list additional
runtime algorithms, but submission still requires a supported execution profile.
Kueue discovery remains unimplemented. Earlier dev-cluster
smoke tests of the broader branch completed small SFT, OSFT, and LoRA SFT
RayJobs with the Training Hub 0.11.2 image
`quay.io/modh/ray@sha256:888ffb8008a095eff0e6f2f97335903f56867773b9493da4a46a92708d54cb55`.
The image reported Ray 2.58.1 despite its 2.58.0 tag; the API used Ray 2.58.0.
LoRA GRPO submitted and reached verl but did not finish due to a TRL/verl
import incompatibility in the runtime image. A compatible image is required
before relying on GRPO training. This image is not confirmed as cluster-shipped
in every target RHOAI release.

