# Algorithm discovery

`GET /api/v1/algorithms` returns compact summaries and `details_url` links.
`GET /api/v1/algorithms/{name}` returns readable parameter types, known defaults,
required parameters, and an editable `example_request` for supported submission profiles.
Copy that complete request body into Swagger's POST endpoint, or save it as JSON:

```bash
curl --fail-with-body -H "Authorization: Bearer $TOKEN" \
  "$TRAINING_SERVICE_URL/api/v1/algorithms/lora_sft" \
  | jq -e '.example_request // error("No example is available for this runtime")' > job.json
```

Edit `job.json` to set a unique job name, model/dataset paths, existing output storage,
and resources appropriate for your model. Add `queue_name` if your project needs a
Kueue LocalQueue, then submit it to
`POST /api/v1/projects/YOUR_PROJECT/training-jobs`. The `training-data` PVC and input
paths in the example are placeholders for project resources; discovery does not
create storage or inspect cluster capacity.

Examples are checked against the configured runtime's parameter schema and the API's
execution-profile validation. OSFT includes required parameters; LoRA GRPO uses a
GPU-free head plus one GPU worker and shared PVC storage. LoRA GRPO also needs a
runtime with compatible verl and TRL packages, which discovery does not verify;
its details include a note saying so. Example values are
illustrative, not runtime defaults, resource recommendations, or a guarantee of
successful training. Native per-parameter examples remain sourced from the runtime.
`example_request` is omitted with a note when no compatible example is available,
including unsupported profiles, Python-only interfaces, and required parameters
without a usable example/default/choice. Discovery never fetches schema references
to construct an example.

Add `?include=parameter_schema` only when the native JSON Schema is needed.
An empty catalog returns `200 {"algorithms": []}`. A missing, malformed or
image-mismatched startup catalog returns `503 backend_unavailable`.
Nonempty catalogs must identify their runtime when `TRAINING_RAY_IMAGE` is configured.
Unknown names return `404 not_found`; invalid `include` returns `400 bad_request`.

## Configure a deployment

The chart defaults to public `quay.io/modh/ray:2.58.0-py312-cu130-th0112` for the
runtime and enables an HTTPS OpenShift Route. Catalog setup is automatic. The
API default is public `quay.io/modh/training-service:0.1.0`; the runtime
override below is optional:

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
metadata. The API additionally requires non-null `num_epochs` and
`effective_batch_size` values to be at least 1, before any storage lookups.
A newer runtime may add parameters without changing the API image.

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

The detail endpoint supplies an `example_request` like the following. Replace the
model and dataset paths, output path, and resources for your environment. The PVC
must exist in the target project and contain the model and algorithm-compatible
JSONL data. Submit this body to
`POST /api/v1/projects/YOUR_PROJECT/training-jobs` with your bearer token.

```json
{
  "name": "lora-sft-example",
  "backend": "ray",
  "model": {"uri": "pvc://training-data/models/my-model/"},
  "dataset": {"uri": "pvc://training-data/datasets/chat.jsonl"},
  "algorithm": {
    "name": "lora_sft",
    "parameters": {
      "backend": "unsloth",
      "learning_rate": 0.00002,
      "num_epochs": 1,
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
  "output": {"destination_uri": "pvc://training-data/outputs/lora-sft-example/"}
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
It also checks that discovered example bodies pass submission validation and reach
the expected storage lookup after only project-specific identifiers are replaced.
Namespace cleanup removes the chart-managed script and generated catalog volume.
See [the E2E runner instructions](../tests/e2e/README.md).
