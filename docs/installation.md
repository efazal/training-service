# Installation and development image workflow

The Training Service is distributed as a Helm chart. The chart installs the
service into an existing OpenShift cluster; it does not install RHOAI, Ray, or
Kueue.

## Prerequisites

- Python 3.11 and `uv` for local checks.
- Podman for image builds.
- Helm 3 for chart validation and installation.
- `kubectl` or `oc` access to the target cluster.
- Push access to a registry that the target cluster can pull from.

## Build and publish a development image

Run the repository checks before building an image:

```bash
make check
```

Build and push an explicitly tagged image. Do not use `latest` for a shared
development or customer handoff:

```bash
export IMAGE_REPOSITORY=quay.io/<your-user>/training-service
export IMAGE_TAG=dev-$(git rev-parse --short HEAD)
export IMAGE_PLATFORM=linux/amd64

make image-build \
  IMAGE_REPOSITORY="$IMAGE_REPOSITORY" \
  IMAGE_TAG="$IMAGE_TAG" \
  IMAGE_PLATFORM="$IMAGE_PLATFORM"
make image-smoke IMAGE_REPOSITORY="$IMAGE_REPOSITORY" IMAGE_TAG="$IMAGE_TAG"
make image-push IMAGE_REPOSITORY="$IMAGE_REPOSITORY" IMAGE_TAG="$IMAGE_TAG"
```

`make image-smoke` starts the just-built image, verifies `/healthz` succeeds and
`/readyz` returns `503` without a startup catalog, and removes the temporary
container. The chart's init container supplies that catalog on deployment.
The Containerfile packages the
runnable FastAPI service from `src/`, including the generated API scaffold from
RHOAIENG-96348. The scaffold registers the contract routes under `/api/v1`; its
unimplemented operations return `501` until their runtime integrations land.

## Validate the chart

```bash
make helm-lint
make helm-template
```

`make helm-template` validates the default chart and an explicit OpenShift
Route/TLS hostname override. It does not install anything on a cluster.

## Install or upgrade from a checkout

Use the exact image tag that was pushed above:

```bash
helm upgrade --install training-service ./charts/training-service \
  --namespace training-service \
  --create-namespace \
  --set-string image.repository="$IMAGE_REPOSITORY" \
  --set-string image.tag="$IMAGE_TAG"
```

The service image and training image are separate settings. The chart defaults
to public `quay.io/modh/ray:2.58.0-py312-cu130-th0112`, matching the API's Ray pin
and including CUDA 13.0 and Training Hub 0.11.2. Override `trainingRuntime.image`
when needed; the chart automatically discovers its algorithms and passes the image to the
service internally. No `TRAINING_RAY_IMAGE` entry under `env` is needed. Legacy
entries still override the chart default; combining one with a non-default
`trainingRuntime.image` is rejected.

The intended public API image default is `quay.io/modh/training-service:0.1.0`.
This image has not been verified publicly accessible: a `modh` maintainer must
create or configure the repository as public and publish the tag before a
default installation can pull it. Until then, use the image overrides above.

For a private registry, create the pull secret first and pass it through chart
values:

```bash
oc get namespace training-service >/dev/null 2>&1 || \
  oc create namespace training-service

oc -n training-service create secret docker-registry training-service-pull \
  --docker-server=quay.io \
  --docker-username='<username>' \
  --docker-password='<token>'

helm upgrade --install training-service ./charts/training-service \
  --namespace training-service \
  --create-namespace \
  --set-string image.repository="$IMAGE_REPOSITORY" \
  --set-string image.tag="$IMAGE_TAG" \
  --set 'imagePullSecrets[0].name=training-service-pull'
```

Prefer a checked-in, environment-specific values file over a long list of
`--set` arguments when configuring replicas, resources, service settings, or
runtime environment variables:

```yaml
# values-development.yaml
replicaCount: 1

image:
  repository: quay.io/example/training-service
  tag: dev-abcdef0

trainingRuntime:
  image: quay.io/YOUR_ORG/ray-traininghub@sha256:YOUR_TESTED_DIGEST

resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 1
    memory: 1Gi

env:
  - name: EXAMPLE_RUNTIME_SETTING
    value: example
```

The stable authentication, authorization, and SDK runtime variable names are
owned by RHOAIENG-96350 through RHOAIENG-96352. Do not add privileged service
account permissions or embed credentials in values while that design is
pending. Use `envFrom` with project-local Secret or ConfigMap references after
those contracts are defined.

## Verify rollout and health

```bash
kubectl -n training-service rollout status deployment/training-service
kubectl -n training-service get deployment,pod,service

export SERVICE_PORT=8080  # Match service.port in the values file.
kubectl -n training-service port-forward service/training-service 8080:"$SERVICE_PORT"
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8080/healthz
curl --fail http://127.0.0.1:8080/readyz
```

These health checks verify the packaged process. Final delivery verification
should also confirm the generated API is available:

```bash
curl --fail http://127.0.0.1:8080/docs
curl --fail http://127.0.0.1:8080/openapi.json
curl -i http://127.0.0.1:8080/api/v1/algorithms
```

The generated API operation currently returns `501` from the OpenShift
authentication stub. This confirms the generated route is registered, but it
does not prove the operation is ready for customer use; authentication and
backend integrations are tracked by RHOAIENG-96350 through RHOAIENG-96352.

## Log-stream lifetime

Each `GET .../logs?follow=true` connection has a total lifetime of 300 seconds
(five minutes) by default, not an idle timeout. For a two-hour job, allow extra
time by setting a three-hour limit in your Helm values:

```yaml
env:
  - name: TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS
    value: "10800"
```

Merge this entry into any existing `env` list and upgrade the release to roll
out the change. The setting is read at application startup and must be a
positive integer number of seconds; invalid values prevent startup. Zero does
not disable the limit. No per-request timeout parameter is required.

At expiry, the API emits an SSE `error` event with code `stream_expired` and
message `Reconnect to continue.`, then closes the connection. Reconnecting
reauthenticates the caller; a longer connection retains its initial
authorization for longer. Training continues unaffected. Snapshot requests
(`follow=false`) are unchanged. Ingress/network timeouts can close a stream
earlier, and this setting does not extend token validity or retain logs after
the Ray cluster is removed.

## OpenShift Route and TLS

An HTTPS Route is enabled by default, with a cluster-assigned hostname and HTTP
redirected to HTTPS. TokenReview and project-scoped authorization remain enabled.
Set `route.enabled=false` for an internal-only Service, or override the hostname:

```yaml
route:
  enabled: true
  host: training-service.apps.example.com
  tls:
    enabled: true
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
```

Install with the values file and inspect the resulting route:

```bash
helm upgrade --install training-service ./charts/training-service \
  --namespace training-service \
  --create-namespace \
  --values values-development.yaml

oc -n training-service get route training-service
```

## Versioned customer handoff

A handoff is a matched pair: a versioned image and a chart whose `appVersion`
identifies that same image version. Package the chart only after the exact image
tag has been built, pushed, and smoke-checked:

```bash
export RELEASE_VERSION=0.1.0
export IMAGE_REPOSITORY=quay.io/modh/training-service
export IMAGE_PLATFORM=linux/amd64

make image-build \
  IMAGE_REPOSITORY="$IMAGE_REPOSITORY" \
  IMAGE_TAG="$RELEASE_VERSION" \
  IMAGE_PLATFORM="$IMAGE_PLATFORM"
make image-smoke \
  IMAGE_REPOSITORY="$IMAGE_REPOSITORY" \
  IMAGE_TAG="$RELEASE_VERSION"
make image-push \
  IMAGE_REPOSITORY="$IMAGE_REPOSITORY" \
  IMAGE_TAG="$RELEASE_VERSION"

make helm-package \
  CHART_VERSION="$RELEASE_VERSION" \
  APP_VERSION="$RELEASE_VERSION"

helm show chart "dist/training-service-${RELEASE_VERSION}.tgz"
```

The handoff record should contain:

- Image reference and immutable digest, for example
  `quay.io/modh/training-service:0.1.0@sha256:<digest>`.
- Chart archive, for example `dist/training-service-0.1.0.tgz`.
- Chart version and `appVersion`.
- Values file used for the verified installation.
- Cluster/RHOAI compatibility and the smoke-check result.

Do not label a pair as customer-ready until the required runtime/security
dependencies are integrated and the exact pair has been verified together.
