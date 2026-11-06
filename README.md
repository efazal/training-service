# OpenDataHub Training Service

Python/FastAPI service generated from the Training API contract in
`docs/api/openapi.yaml`. OpenShift TokenReview authenticates callers,
SubjectAccessReview authorizes project operations, and POST training-job
submission uses CodeFlare. Job listing, details, driver logs, and terminal
RayJob deletion and algorithm discovery are implemented. Queue discovery and
estimation remain fail-closed with `501`.

## Ray submission and algorithm discovery

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

```yaml
image:
  repository: quay.io/modh/training-service
  tag: "0.1.0"
trainingRuntime:
  image: quay.io/modh/ray:2.58.0-py312-cu130-th0112
route:
  enabled: true
```

These are the chart defaults. The HTTPS OpenShift Route gets a cluster-assigned
hostname. Set `route.enabled=false` for an internal-only deployment.
The service repository above is the intended public release location, but a
public image there has not been verified. A `modh` maintainer must create or
configure the repository as public and publish the service image first. Until
then, override `image.repository` and `image.tag` with an available image.

Legacy values files using `env: [{name: TRAINING_RAY_IMAGE, value: ...}]`
still override the default. Move that entry to `trainingRuntime.image`; combining
a non-default runtime with the legacy entry is rejected.

`pvc://` URIs refer to project PVCs. `hf://` and `s3://` URIs refer to
project Secrets; the selected training image also needs `datasets` and
`huggingface_hub` for the corresponding storage paths. S3 submissions
preflight `boto3` and `botocore`; if either is missing, the Ray driver attempts
to install a pinned pair into its writable job-local directory before training.
This requires a reachable package index or mirror; otherwise the job fails
early with an S3 dependency error. Resource
lookups and RayJob creation use the verified caller's token, while credentials
are mounted into Ray pods rather than embedded in the RayJob.

An opt-in S3 upload check at `tests/integration/test_s3_output.py` writes a
checkpoint object, reads it back, and deletes it. Set `TRAINING_S3_E2E_ENDPOINT`,
`TRAINING_S3_E2E_BUCKET`, `TRAINING_S3_E2E_ACCESS_KEY`, and
`TRAINING_S3_E2E_SECRET_KEY`, then run
`uv run --with 'boto3==1.40.61' pytest tests/integration/test_s3_output.py`.

See [job operations](docs/job-operations.md) for permissions, log gateway
configuration, response examples, and cascading cleanup behavior.

## Local development

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a running
Podman engine for OpenAPI Generator.

```bash
make install       # generate server, then install dependencies
make check         # lint, formatting, typing, focused unit tests
make run
```

Open http://localhost:8080/docs. Probes are at `/healthz` and `/readyz`.

See [live E2E tests](tests/e2e/README.md) for `make test-e2e` against an existing
service, or `make test-e2e-deploy` to build with Podman, push to Quay, create a
fresh Kueue-labeled namespace, deploy Helm and test real queued training in one
command. Ray and Kueue must already be installed/configured; direct installations
are supported without checking DSC management state. Kueue failures are reported
without stopping independent API tests. Live tests are excluded from unit-test/CI defaults.

Generated code lives in the committed `src/training_service_api/` package.
Its README and every Python file identify it as generated and not to be edited
directly. Edit the contract or templates, run `make generate-server`, and
commit the updated sources. Handwritten behavior belongs in
`src/training_service/api_impl/` and supporting modules.

Generate before building the existing container:

```bash
make generate-server
podman build -f Containerfile -t training-service:dev .
```

The Python wheel and container include the generated package and canonical
contract. Helm/deployment changes and backend implementations are separate
tasks.

See [API generation](docs/api/README.md) and [structure](docs/architecture.md).
