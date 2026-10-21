# OpenDataHub Training Service

Python/FastAPI service generated from the Training API contract in
`docs/api/openapi.yaml`. OpenShift TokenReview authenticates callers,
SubjectAccessReview authorizes project operations, and POST training-job
submission uses CodeFlare. Other API operations fail closed with `501`.

## Basic Ray submission

POST supports Training Hub `lora_sft` with the four basic numeric parameters
in the contract. It maps resources and queue settings to a CodeFlare RayJob.
The `202` response returns the RayJob name as `id` and a project-scoped
`Location` header; it does not mean training has completed.

Set `TRAINING_RAY_IMAGE` through the chart's existing `env` values or the
process environment. The image must contain Ray and Training Hub, with a Ray
version matching the version CodeFlare writes into the RayJob. There is no
default image; submission returns `503` until one is configured. The API
container bundles a pinned CodeFlare Git revision for caller-scoped SDK
submission. No service-side job store or hand-authored RayJob manifest is used.

`pvc://` URIs refer to project PVCs. `hf://` and `s3://` URIs refer to
project Secrets; the selected training image also needs `datasets`,
`huggingface_hub`, and `boto3` for the corresponding storage paths. Resource
lookups and RayJob creation use the verified caller's token, while credentials
are mounted into Ray pods rather than embedded in the RayJob.

## Local development

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a running
Podman or Docker engine for OpenAPI Generator.

```bash
make install       # generate server, then install dependencies
make check         # lint, formatting, typing, focused unit tests
make run
```

Open http://localhost:8080/docs. Probes are at `/healthz` and `/readyz`.

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
