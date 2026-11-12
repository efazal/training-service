# Training Service Helm chart

Install on a prepared OpenShift AI cluster with Ray/Kueue. The installer needs
permission to create the chart's authentication and platform-discovery RBAC.

```bash
helm repo add training-service https://raw.githubusercontent.com/opendatahub-io/training-service/helm-charts/
helm repo update training-service
helm install training-service training-service/training-service \
  --namespace training-service --create-namespace --wait --timeout 15m
```

Defaults use public `quay.io/modh/training-service:0.1.0` and the configured
Ray/Training Hub runtime. The chart creates an HTTPS Route and discovers the
platform log gateway and ingress CA. No values file, registry login or manually
created CA ConfigMap is required.

To customize, run `helm pull training-service/training-service --untar`, edit
`training-service/values.yaml`, then install `./training-service` instead.

The Helm output shows how to get the API URL. Open `/docs` there; API requests
require an OpenShift bearer token. Training jobs need a project, compatible
model/dataset storage and sufficient GPU/queue capacity.
