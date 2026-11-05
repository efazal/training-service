# Algorithm discovery

`GET /api/v1/algorithms` returns compact summaries and `details_url` links.
`GET /api/v1/algorithms/{name}` returns readable parameter types, known defaults,
required parameters, and examples only when supplied by the runtime schema.
Discovery does not generate POST requests or choose models, storage, or resources.
Add `?include=parameter_schema` only when the native JSON Schema is needed.
An empty configured catalog returns `200 {"algorithms": []}`. An unconfigured,
missing, malformed or image-mismatched catalog returns `503 backend_unavailable`.
Nonempty catalogs must identify their runtime when `TRAINING_RAY_IMAGE` is configured.
Unknown names return `404 not_found`; invalid `include` returns `400 bad_request`.

## Configure a deployment

An administrator creates a ConfigMap containing `catalog.json` from the **exact
runtime image** used for training. The API reads it directly through Kubernetes
using the authenticated caller's bearer token, never CodeFlare/Ray or elevated
service-account credentials. Callers need `get` permission on that ConfigMap in
its namespace. Kubernetes authentication/authorization failures remain 401/403.
The chart does not grant this permission automatically.

Run the exporter inside the selected runtime image, without a GPU:

```bash
RUNTIME_IMAGE=quay.io/YOUR_ORG/runtime@sha256:YOUR_DIGEST
podman run --rm --entrypoint python -v "$PWD/scripts:/scripts:ro" \
  "$RUNTIME_IMAGE" /scripts/export_algorithm_catalog.py "$RUNTIME_IMAGE" > catalog.json
oc create configmap training-algorithms --from-file=catalog.json -n YOUR_NAMESPACE
helm upgrade --install training-service charts/training-service -n YOUR_NAMESPACE \
  -f YOUR_VALUES.yaml --set algorithmCatalog.name=training-algorithms \
  --set-string trainingRuntime.image="$RUNTIME_IMAGE"
```

`algorithmCatalog.namespace` defaults to the Helm release namespace. Without Helm,
set `TRAINING_SERVICE_CATALOG_NAME` and `TRAINING_SERVICE_CATALOG_NAMESPACE`.
Do not also set those environment variables when using `algorithmCatalog` values.
Keep Helm's `trainingRuntime.image` and the catalog's `runtime.image` identical.
The chart sets the internal `TRAINING_RAY_IMAGE` environment variable for the service.
For non-Helm deployments, configure that environment variable directly. Refresh the
ConfigMap when the runtime changes. No catalog cache or service restart is needed.

Use a namespace-local Role granting `get` on `configmaps`, with
`resourceNames: [training-algorithms]`, and bind it to the intended users/groups.
Discovery describes the image registry; it does not enable new submission
algorithms. This branch retains basic `lora_sft` submission support. Native
schemas remain intact, and Python callbacks are not JSON-submittable.

## Example training request

This is documentation, not a generated discovery response or a recommended
configuration. Replace the model and dataset paths, output path, and resources
for your environment. The PVC must exist in the target project and contain the
model and algorithm-compatible JSONL data. Submit this body to
`POST /api/v1/projects/YOUR_PROJECT/training-jobs` with your bearer token.

```json
{
  "name": "lora-sft-example",
  "backend": "ray",
  "model": {"uri": "pvc://YOUR_PVC/models/YOUR_MODEL/"},
  "dataset": {"uri": "pvc://YOUR_PVC/datasets/train.jsonl"},
  "algorithm": {
    "name": "lora_sft",
    "parameters": {
      "learning_rate": 0.0001,
      "num_epochs": 2,
      "effective_batch_size": 8,
      "max_seq_len": 2048
    }
  },
  "resources": {
    "num_nodes": 1,
    "gpus_per_node": 1,
    "gpu_type": "nvidia.com/gpu",
    "cpus_per_node": "8",
    "memory_per_node": "32Gi"
  },
  "output": {"destination_uri": "pvc://YOUR_PVC/outputs/lora-sft-example/"}
}
```

Add `queue_name` if your project requires a Kueue LocalQueue. The values above
are illustrative, not algorithm defaults or a guarantee of GPU/model fit.

## Verification

`make test` runs mocked Kubernetes unit tests; no cluster is needed.
`make test-e2e-deploy` also exports the selected runtime's registry into a
test-owned ConfigMap before Helm installation. It checks compact responses,
details/schema links, native parameter examples, unknown names and authentication, alongside the
existing queued training lifecycle. Namespace cleanup removes the catalog too.
See [the E2E runner instructions](../tests/e2e/README.md).
