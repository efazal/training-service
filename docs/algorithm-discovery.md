# Algorithm discovery

`GET /api/v1/algorithms` returns compact summaries and `details_url` links.
`GET /api/v1/algorithms/{name}` returns readable parameter types, known defaults,
required parameters, and examples only when supplied by the runtime schema.
Discovery does not generate POST requests or choose models, storage, or resources.
Add `?include=parameter_schema` only when the native JSON Schema is needed.
An empty catalog returns `200 {"algorithms": []}`. A missing, malformed or
image-mismatched startup catalog returns `503 backend_unavailable`.
Nonempty catalogs must identify their runtime when `TRAINING_RAY_IMAGE` is configured.
Unknown names return `404 not_found`; invalid `include` returns `400 bad_request`.

## Configure a deployment

The chart defaults to public `quay.io/modh/ray:2.58.0-py312-cu130-th0112` for the
runtime and enables an HTTPS OpenShift Route. Catalog setup is automatic. The
intended API default, `quay.io/modh/training-service:0.1.0`, still needs verified
public publication by a `modh` maintainer. Until then, supply an available API
image; the runtime override below is optional:

```bash
helm upgrade --install training-service charts/training-service -n YOUR_NAMESPACE \
  --create-namespace \
  --set-string image.repository=quay.io/YOUR_ORG/training-service \
  --set-string image.tag=YOUR_SERVICE_TAG
```

To use another compatible runtime, add
`--set-string trainingRuntime.image=quay.io/modh/ray@sha256:YOUR_TESTED_DIGEST`.

1. Kubernetes starts `discover-algorithms`, an init container using that same runtime
   image. The chart supplies the exporter; it imports Training Hub and inspects the
   installed registry, public callables, types, required arguments and defaults.
   No GPU, model, dataset, Ray cluster or Kueue admission is needed for discovery.
2. The exporter atomically writes `catalog.json` into a shared `emptyDir` volume.
   A chart-managed ConfigMap contains only the exporter script, not a manual catalog.
3. The API starts and caches the validated file. It mounts the catalog read-only.
   Discovery and POST submission use the same metadata; requests do not contact
   Kubernetes to read it. Callers still authenticate normally, but need no catalog
   ConfigMap permissions. No additional service-account workload permissions are added.
4. Changing `trainingRuntime.image` rolls out the pods and regenerates the catalog.
   Use an immutable image digest so later training pods cannot pull different code
   under the same mutable tag. Each API replica generates its own startup catalog.

Existing `imagePullSecrets` also cover the init container when the runtime is private.
The runtime image must match the cluster node architecture. Initial deployment can
take longer while Kubernetes pulls the training image.

Unsupported algorithms/parameters, incorrect types, missing required parameters and
invalid choices return `422 validation_error` before any workload resources are created.
Defaults remain owned by Training Hub. Python-only interfaces and callbacks are not
JSON-submittable. Schema checks do not guarantee dataset compatibility, GPU fit or
successful execution; model/data/storage and backend requirements still apply.
The merged execution profiles remain `sft`, `osft`, `lora_sft`, and `lora_grpo`.
An algorithm being present in Training Hub does not add an execution profile;
runtime-only algorithms without one return `422` before Kubernetes access.
Adapter checks for topology and incompatible parameter combinations still run,
but no static parameter allowlist or fallback schema is used. Signature-derived
schemas describe types/defaults, not numerical bounds absent from the runtime
metadata. A newer runtime may add parameters without changing the API image.

If export fails, Kubernetes will not start the API container. Inspect:

```bash
oc logs -n YOUR_NAMESPACE deployment/training-service -c discover-algorithms
```

A missing, invalid or mismatched file keeps `/readyz` at `503`; `/healthz` and `/docs`
remain available if the API process has started. No silent hardcoded-catalog fallback
is used. The former `algorithmCatalog` values and catalog-name/namespace environment
variables are no longer needed; remove them from old values files.

For local development only, run `charts/training-service/files/export_algorithm_catalog.py`
inside the runtime, saving stdout to a file, and set `TRAINING_SERVICE_CATALOG_PATH`
and `TRAINING_RAY_IMAGE` before starting the API. Helm sets these automatically.

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

`make test` verifies startup export, readiness, runtime-specific validation, dispatch,
and mocked Kubernetes operations; no cluster is needed.
`make test-e2e-deploy` exercises the normal chart's init container, without preparing
a separate catalog. It checks compact responses, details/schema links, incompatible
submissions, unknown names and authentication alongside the queued training lifecycle.
Namespace cleanup removes the chart-managed script and generated catalog volume.
See [the E2E runner instructions](../tests/e2e/README.md).
