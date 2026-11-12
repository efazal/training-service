# OpenDataHub Training Service

Fine-tune models through a project-scoped REST API on Red Hat OpenShift AI.

Fine-tuning often starts with a workbench, a suitable notebook image, a training
loop, and a backend SDK. That flexibility is useful for advanced workflows,
but creates extra setup for applications that need a straightforward way to
adapt a model to their data.

Training Service exposes Training Hub recipes through one HTTP interface and
handles backend-specific job submission. Dashboards, CLIs, and other applications
can choose a model, dataset, algorithm, resources, and output destination without
requiring their users to configure a notebook or learn a training SDK. Existing
workbench and SDK workflows remain available for advanced customization.

**RHOAI 3.6 Developer Preview supports Ray as the execution backend.** The API
handles that integration; callers submit training configuration and monitor jobs
through the same service URL. OpenShift bearer tokens and project permissions
control access. Every new job requires an S3 or PVC output destination so trained
model artifacts remain available after job cleanup.

- **Install the service:** [Helm quickstart](docs/installation.md).
- **Integrate a dashboard:** [API quickstart](docs/dashboard-integration.md).
- **Explore the API:** open `/docs` on your service, or read the upstream
  [OpenAPI contract](https://github.com/opendatahub-io/training-service/blob/main/docs/api/openapi.yaml).

Supported submission profiles are `sft`, `osft`, `lora_sft`, and `lora_grpo`.
Discovery and submission validation use the same catalog from the configured
training image. Check [runtime compatibility](docs/training-runtime.md) before
selecting an image; GRPO requires a compatible verl/TRL runtime. The queue
discovery endpoint is reserved for listing project LocalQueues with Kueue enabled
in RHOAI; it currently returns `501` until implemented.

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
