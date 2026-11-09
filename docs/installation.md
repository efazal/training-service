# Install Training Service

For platform administrators. Use an existing OpenShift/RHOAI cluster with its
training backend and authenticated gateway configured, Helm 3, `oc`, and `uv`.
The installer needs namespace/Helm permissions and permission to create the
TokenReview/SubjectAccessReview RBAC in this chart. The chart does not install
the platform or grant workload permissions to API callers.

## Install

From the repository checkout, while logged into the target cluster:

```bash
uv sync --locked
uv run python scripts/platform_values.py > /tmp/training-service-values.yaml

helm upgrade --install training-service ./charts/training-service \
  --namespace training-service \
  --create-namespace \
  --values /tmp/training-service-values.yaml \
  --wait --timeout 15m

export TRAINING_SERVICE_URL="https://$(oc -n training-service get route training-service -o jsonpath='{.spec.host}')"
```

The helper reads the platform gateway and **public** ingress CA and generates
values; it makes no cluster changes. Helm mounts the CA and discovers algorithms
from the training image automatically. Share `TRAINING_SERVICE_URL` and the
[dashboard quickstart](dashboard-integration.md) with the integration team.
Swagger UI is at `$TRAINING_SERVICE_URL/docs`.

The default API image is the public amd64 developer preview
`quay.io/modh/training-service:preview-93228d235ee8`. It contains the log-timeout and active-job
deletion changes. This is a preview build, not a stable release. Keep the API
image and chart from the same handoff. The separate `trainingRuntime.image`
selects the training image; see [compatibility](training-runtime.md).

## Platform settings

For platforms without automatic discovery, create a values override:

```yaml
logBackend:
  gatewayURL: https://gateway.apps.example.com
  streamTimeoutSeconds: 300
  # Optional when normal container trust already verifies the gateway:
  trustedCA:
    configMapName: platform-gateway-ca
    key: ca-bundle.crt
```

The gateway is the platform's authenticated backend entry point, **not** the
Training Service URL shared with API callers. A custom CA is needed only when
its issuer is absent from the container trust store. `trustedCA.configMapName`
references a ConfigMap in the service namespace. Alternatively, `trustedCA.bundle`
accepts public PEM certificates, as produced by the helper. TLS verification
stays enabled. Use the helper's `--system-trust` option to omit custom trust.

For a platform-managed ConfigMap, new connections read mounted CA updates.
For the helper's Helm-managed bundle, rerun the helper and Helm upgrade after
certificate rotation. API clients must also trust the platform's Route issuer;
provide the normal platform CA through your organization's trust setup.

Helm rejects missing or conflicting settings before installation. `/readyz`
checks the catalog and resolved gateway/CA configuration without needing a job;
`/healthz` checks only the process. Readiness does not probe backend reachability.
Legacy gateway/CA environment settings still work; remove them when adopting
structured settings. Keep shared values free of credentials and cluster-specific
URLs. Set `route.enabled=false` when exposing the internal Service through your
platform's own proxy.

## Build or share a chart

Maintainers with Quay Write access can publish a commit-specific image:

```bash
podman login quay.io
export IMAGE_TAG=dev-$(git rev-parse --short HEAD)
make generate-server CONTAINER_RUNTIME=podman
make check helm-lint helm-template
make image-build CONTAINER_RUNTIME=podman IMAGE_PLATFORM=linux/amd64 IMAGE_TAG="$IMAGE_TAG"
# Run the image smoke check on an amd64 host.
make image-smoke CONTAINER_RUNTIME=podman IMAGE_TAG="$IMAGE_TAG"
make image-push CONTAINER_RUNTIME=podman IMAGE_TAG="$IMAGE_TAG"
make helm-package
```

These targets use the existing `Containerfile` and `quay.io/modh/training-service`.
For amd64 images built on an ARM workstation, verify startup on amd64 OpenShift
nodes. To deploy your new tag, add `--set-string image.tag="$IMAGE_TAG"` to the
install. Packaging preserves the checked-in preview defaults; include any custom
`image.tag` in your handoff values. Share the chart archive in `dist/`, image
tag/digest, this guide, and platform values through your team's normal handoff channel.
Private image overrides can use `imagePullSecrets`; the default image is public.
