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

- **Install and find the API URL:** [Installation guide](docs/installation.md).
- **Use or integrate the API:** [Integration quickstart](docs/dashboard-integration.md).
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

### Build and publish

Maintainers with Quay Write access can publish a commit-specific image from a
source checkout:

```bash
podman login quay.io
export IMAGE_REPOSITORY=quay.io/modh/training-service
export IMAGE_TAG=dev-$(git rev-parse --short HEAD)
make generate-server CONTAINER_RUNTIME=podman
make check helm-lint helm-template
make image-build CONTAINER_RUNTIME=podman IMAGE_PLATFORM=linux/amd64 IMAGE_TAG="$IMAGE_TAG"
# Run the image smoke check on an amd64 host.
make image-smoke CONTAINER_RUNTIME=podman IMAGE_TAG="$IMAGE_TAG"
make image-push CONTAINER_RUNTIME=podman IMAGE_TAG="$IMAGE_TAG"
make helm-package
```

These targets use the existing `Containerfile`. For an amd64 image built on an
ARM workstation, verify startup on amd64 OpenShift nodes. Packaging preserves the
checked-in defaults; edit the chart's `image.tag` before packaging if the handoff
should use your new image. For handoff, share the
[installation guide](docs/installation.md) and the API URL. Include the archive
from `dist/` and its matching image tag/digest when distributing an unpublished chart.
