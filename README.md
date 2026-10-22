# OpenDataHub Training Service

Python/FastAPI service generated from the Training API contract in
`docs/api/openapi.yaml`. OpenShift TokenReview authenticates callers,
SubjectAccessReview authorizes project operations, and POST training-job
submission uses CodeFlare. Job listing, details, driver logs, and terminal
RayJob deletion are implemented. Algorithm/queue discovery and estimation
remain fail-closed with `501`.

## Basic Ray submission

POST supports Training Hub `lora_sft` with the four basic numeric parameters
in the contract. It maps resources and queue settings to a CodeFlare RayJob.
The `202` response returns the RayJob name as `id` and a project-scoped
`Location` header; it does not mean training has completed.

Set `TRAINING_RAY_IMAGE` through the chart's existing `env` values or the
process environment. The image must contain Ray and Training Hub, with a Ray
version matching the version CodeFlare writes into the RayJob. The tested
combination is CodeFlare SDK 0.38.2 with Ray 2.55.1 in the API image and
`quay.io/modh/ray:2.55.1-py312-cu129-th081` for the RayJob (Ray 2.55.1,
Training Hub 0.8.1). Use an approved image digest for deployment; this Quay
tag has not been confirmed as shipped with RHOAI 3.6 GA. There is no default
training image; submission returns `503` until one is configured. The API
container installs the RHOAI 3.6 CodeFlare and Ray wheels. CodeFlare 0.38.2
uses process-global authentication, so the adapter serializes SDK submission
and restores the prior client after each caller. No service-side job store or
hand-authored RayJob manifest is used.

`pvc://` URIs refer to project PVCs. `hf://` and `s3://` URIs refer to
project Secrets; the selected training image also needs `datasets`,
`huggingface_hub`, and `boto3` for the corresponding storage paths. Resource
lookups and RayJob creation use the verified caller's token, while credentials
are mounted into Ray pods rather than embedded in the RayJob.

See [job operations](docs/job-operations.md) for permissions, log gateway
configuration, response examples, and cascading cleanup behavior.

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
