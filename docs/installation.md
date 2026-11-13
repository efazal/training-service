# Install and access Training Service

Start here to install the service, find its HTTPS address, and make your first
API request. This is the guide to share with a team receiving the Helm chart.
RHOAI 3.6 Developer Preview supports Ray as the training backend.

## Before you start

- Use an existing OpenShift cluster running **RHOAI 3.5 or later**, with Ray and
  its authenticated gateway configured.
- Install Helm 3 and `oc`, and log into the target cluster with `oc login`.
- Have a platform administrator install the chart. It creates cluster-scoped RBAC
  for authentication and gateway discovery, plus a Role in `openshift-config-managed`.
  API callers use their own project permissions.

The chart installs Training Service. It does not install Ray or prepare model
and dataset storage.

## 1. Install the chart

```bash
helm repo add training-service https://raw.githubusercontent.com/opendatahub-io/training-service/helm-charts/
helm repo update training-service
helm upgrade --install training-service training-service/training-service \
  --namespace training-service \
  --create-namespace \
  --wait --timeout 15m
```

No values file is needed for the standard installation. Gateway, public ingress
CA, and algorithm discovery are automatic. The first install downloads a training
image larger than 10 GB. If Helm times out while it is downloading, wait for the
pod to become ready and rerun the same install command.

## 2. Find the API URL

The chart creates an HTTPS address automatically. OpenShift calls this a
**Route**; you do not need to create one or choose a hostname.

```bash
export TRAINING_SERVICE_URL="https://$(oc -n training-service get route training-service -o jsonpath='{.spec.host}')"
printf 'API URL: %s\nAPI reference: %s/docs\n' "$TRAINING_SERVICE_URL" "$TRAINING_SERVICE_URL"
```

The API URL looks like `https://training-service-training-service.apps.example.com`.
Share this address with API users. For a customized release name or namespace,
retrieve the matching lookup command with
`helm get notes RELEASE_NAME -n NAMESPACE`.

## 3. Open the API and check access

Open the printed **API reference** URL in your browser. This is Swagger UI, where
you can browse endpoints and try requests. Select **Authorize** and enter your
OpenShift access token (`oc whoami -t`) to call protected endpoints.

From a terminal logged into OpenShift, check authenticated access:

```bash
export TOKEN="$(oc whoami -t)"
curl --fail-with-body -H "Authorization: Bearer $TOKEN" \
  "$TRAINING_SERVICE_URL/api/v1/algorithms"
```

A successful response returns JSON listing the available algorithms. API requests
start with `$TRAINING_SERVICE_URL/api/v1`. Clients must trust your organization's
OpenShift HTTPS certificates.

To submit a job or connect an application, continue with the
[API integration quickstart](dashboard-integration.md). It covers authentication,
storage, submission, status, logs, and deletion. The optional
[runtime reference](training-runtime.md) explains training image compatibility and
algorithm requirements.

## Optional installation settings

### Customize the chart

Download the chart, edit its `values.yaml`, then install the local copy:

```bash
helm pull training-service/training-service --untar
# Edit training-service/values.yaml, then:
helm upgrade --install training-service ./training-service \
  --namespace training-service \
  --create-namespace \
  --wait --timeout 15m
```

### Container images

Training Service images are available in
[quay.io/modh/training-service](https://quay.io/repository/modh/training-service?tab=tags).
The chart defaults to `quay.io/modh/training-service:0.1.0`. Keep the API image
and chart from the same release.

Training runtime images are available in
[quay.io/modh/ray](https://quay.io/repository/modh/ray?tab=tags), with these three tags:

| Runtime image | Selection |
| --- | --- |
| `quay.io/modh/ray:2.58.0-py312-cu130-th0112` | Chart default |
| `quay.io/modh/ray:2.58.0-py312-cu129-th081` | Alternative |
| `quay.io/modh/ray:2.55.1-py312-cu129-th081` | Alternative |

To change images, edit `image.repository` and `image.tag` for the API service,
or `trainingRuntime.image` for training jobs, in the chart's `values.yaml`.
Check [runtime compatibility](training-runtime.md) before selecting an alternative;
the runtime must be compatible with the API's Ray and SDK versions.
Private images can use `imagePullSecrets`.

### Custom gateway or certificates

For platforms without automatic discovery, edit these settings in the chart's
`values.yaml`:

```yaml
logBackend:
  gatewayURL: https://gateway.apps.example.com
  streamTimeoutSeconds: 300
  trustedCA:
    configMapName: platform-gateway-ca
    key: ca-bundle.crt
```

The gateway is the platform's training-backend address; callers use the API URL
from step 2. A custom CA is needed only if normal container trust cannot verify
the gateway. `configMapName` references an existing ConfigMap in the service
namespace. Alternatively, `trustedCA.bundle` supplies public PEM certificates for
Helm to create one. TLS verification stays enabled.

New connections read mounted ConfigMap updates. Update a Helm-managed bundle and
run Helm upgrade after certificate rotation. The optional
`uv run python scripts/platform_values.py` helper in a source checkout prepares
public CA settings; `--system-trust` omits custom trust. Use structured settings
instead of legacy gateway/CA environment overrides.

### Access without an OpenShift Route

If your platform uses its own proxy, set `route.enabled=false` and get the API
address from the platform administrator. For local access, keep this command
running:

```bash
oc -n training-service port-forward service/training-service 8080:8080
```

Open `http://localhost:8080/docs`. In a second terminal, use
`export TRAINING_SERVICE_URL=http://localhost:8080` for API commands. This local
address is available only while port-forwarding is running.

### Troubleshoot access

- `/readyz` checks the runtime catalog; `/healthz` checks the process. Neither
  checks the training gateway.
- `503 logs_unavailable` on log requests can indicate a gateway or CA problem.
- Helm rejects conflicting settings before installation.
