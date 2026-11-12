# Install Training Service

For platform administrators. Use an existing OpenShift/RHOAI cluster with its
training backend and authenticated gateway configured, Helm 3, `oc`, and `uv`.
The installer needs namespace/Helm permissions and permission to create the
TokenReview/SubjectAccessReview RBAC in this chart. The chart does not install
the platform or grant workload permissions to API callers.

## Install

From the repository checkout, while logged into the target cluster:

```bash
cat > /tmp/training-service-values.yaml <<EOF
env:
  - name: TRAINING_SERVICE_RAY_GATEWAY_URL
    value: https://$(oc get gatewayconfig default-gateway -o jsonpath='{.status.domain}')
EOF

helm upgrade --install training-service ./charts/training-service \
  --namespace training-service \
  --create-namespace \
  --values /tmp/training-service-values.yaml \
  --wait --timeout 15m

export TRAINING_SERVICE_URL="https://$(oc -n training-service get route training-service -o jsonpath='{.spec.host}')"
```

For a shared chart archive, use `./training-service-0.1.1.tgz` in place of the
chart directory.

The values file points the service at the platform gateway. Helm discovers
algorithms from the training image automatically. The first install pulls that
image (over 10 GB); on a fresh node Helm can report `Progress deadline exceeded`
while the pull continues. Wait for the pod to become ready and rerun the same
`helm upgrade --install` command. Share `TRAINING_SERVICE_URL` and the
[dashboard quickstart](dashboard-integration.md) with the integration team.
Swagger UI is at `$TRAINING_SERVICE_URL/docs`.

The default API image is the public amd64 developer preview
`quay.io/modh/training-service:preview-83a2a358fa0c`. This is a preview build, not a stable release. Keep the API
image and chart from the same handoff. The separate `trainingRuntime.image`
selects the training image; see [compatibility](training-runtime.md).

## Platform settings

For a gateway whose certificate is not publicly trusted, or to change the log
stream limit, extend the values file:

```yaml
env:
  - name: TRAINING_SERVICE_RAY_GATEWAY_URL
    value: https://gateway.apps.example.com
  - name: TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS
    value: "300"
  # Optional when normal container trust already verifies the gateway:
  - name: TRAINING_SERVICE_RAY_CA_FILE
    value: /var/run/training-service/ray-ca/ca.crt
volumes:
  - name: ray-gateway-ca
    configMap:
      name: platform-gateway-ca
volumeMounts:
  - name: ray-gateway-ca
    mountPath: /var/run/training-service/ray-ca
    readOnly: true
```

The gateway is the platform's authenticated backend entry point, **not** the
Training Service URL shared with API callers. A custom CA is needed only when
its issuer is absent from the container trust store. `platform-gateway-ca` is a
ConfigMap in the service namespace with key `ca.crt`, for example copied from
`openshift-config-managed/default-ingress-cert`. TLS verification stays enabled.

New connections read mounted CA updates; refresh the ConfigMap after certificate
rotation. API clients must also trust the platform's Route issuer; provide the
normal platform CA through your organization's trust setup.

`/readyz` checks the runtime catalog; `/healthz` checks only the process.
Neither probes the gateway: a wrong gateway URL or untrusted certificate returns
`503 logs_unavailable` on log requests. Keep checked-in defaults free of credentials and cluster-specific
URLs. Set `route.enabled=false` when exposing the internal Service through your
platform's own proxy.

## Build or share a chart

Maintainers with Quay Write access can publish a commit-specific image:

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

These targets use the existing `Containerfile` and `quay.io/modh/training-service`.
For amd64 images built on an ARM workstation, verify startup on amd64 OpenShift
nodes. To deploy your new tag, add `--set-string image.tag="$IMAGE_TAG"` to the
install. Packaging preserves the checked-in preview defaults; include any custom
`image.tag` in your handoff values. Share the chart archive in `dist/`, image
tag/digest, this guide, and platform values through your team's normal handoff channel.
Private image overrides can use `imagePullSecrets`; the default image is public.
