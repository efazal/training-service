# OpenDataHub Training Service

A project-scoped API to discover training algorithms, submit jobs, check status,
read driver logs, and delete jobs. OpenShift bearer tokens and project permissions
control access. Dashboard callers use one service URL; platform administrators
configure the training backend during installation.

- **Install the service:** [Helm quickstart](docs/installation.md).
- **Integrate a dashboard:** [API quickstart](docs/dashboard-integration.md).
- **Explore the API:** open `/docs` on your service, or read the
  [OpenAPI contract](docs/api/openapi.yaml).

Supported submission profiles are `sft`, `osft`, `lora_sft`, and `lora_grpo`.
Algorithm discovery and submission validation use the same catalog, discovered
from the configured training image. Check
[runtime compatibility](docs/training-runtime.md) before selecting an image;
GRPO still requires a compatible verl/TRL runtime. Queue discovery and estimation
return `501`.

## Development

Use Python 3.11+, [uv](https://docs.astral.sh/uv/), Helm 3, and Podman.

```bash
make install CONTAINER_RUNTIME=podman
make check helm-lint helm-template
make run
```

`/healthz` checks the process; `/readyz` also checks its runtime catalog. Use the
Helm quickstart for a configured deployment.

Edit the contract/templates, run `make generate-server CONTAINER_RUNTIME=podman`,
and commit the generated `src/training_service_api/` changes. Handwritten behavior
belongs in `src/training_service/`.

See [API generation](docs/api/README.md), [job operations](docs/job-operations.md),
[algorithm discovery](docs/algorithm-discovery.md), [architecture](docs/architecture.md),
and [live E2E tests](tests/e2e/README.md).
