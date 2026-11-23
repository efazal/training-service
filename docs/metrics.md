# Training metrics in MLflow

When the managed MLflow server is available, Training Service automatically creates
one run for each submission. Runs remain in MLflow after training finishes and the
training job is deleted. MLflow retention and storage policies still apply.

## Submit and view a run

Add an optional experiment name to your usual submission:

```json
{
  "mlflow_experiment_name": "model-fine-tuning"
}
```

This is an additional field, not a complete training request. If omitted, the
service uses the `Default` experiment. The workspace is always the project in
`/api/v1/projects/{project}`; callers cannot select another workspace or server.
An explicitly named experiment requires tracking to be available. Omitting the name
selects `Default`; it does not opt out of tracking or its project permissions.
If the selected experiment was deleted in MLflow, submission returns
`409 mlflow_experiment_deleted`. Restore the experiment in MLflow or select another name.

GET the job as usual. When tracking is enabled, its response includes
`mlflow_url` and `mlflow_run_id`. Open the URL to see the run in MLflow. Save the
link before deleting the job: the job endpoint returns 404 after deletion, while
the MLflow run remains available.

## What is recorded

| Data | Coverage |
| --- | --- |
| Job identity and requested resources | Every supported execution profile |
| Execution start/end, duration and success/failure | Driver lifecycle, including input loading and output upload |
| Training start/end, epoch, step, loss, learning rate | Algorithms whose runtime exposes unified callbacks |
| Evaluation and backend numeric metrics | Available callback metrics, including nested numeric values |
| Checkpoint time, step, count and path | Successful saves reported through `on_save` |
| Checkpoint size | Backend-reported size, or a confined, no-follow scan inside the approved output tree while its identity remains unchanged |
| Checkpoint save duration and type | Only when explicitly reported by the backend |
| Pod CPU time, throttling, memory and storage I/O | Cgroup v2 counters from instrumented pods |
| NVIDIA GPU utilization, used memory and power | Samples restricted to explicitly assigned devices visible to each instrumented process |
| Per-rank process CPU, peak RSS and CUDA allocator peaks | Callback-enabled distributed trainers |
| Final numeric results and output directory size | When returned by the execution profile |

Training metrics use `train/`, evaluation metrics `eval/`, checkpoint events
`checkpoint/`, and resource samples `system/pod_<name>/`. Inspect
`tracking.unified_callbacks`, `checkpoint.duration_source`, and
`tracking.delivery` to see the available coverage and delivery health. The driver
aggregates `tracking.producer.*` summaries and verifies the expected trainer ranks
before marking delivery complete. `tracking/failed_events` and
`tracking/unresolved_events` distinguish failed/dropped writes from work still
pending at the flush deadline. Events are batched with their original timestamps
and steps. NaN or infinity is omitted from numeric payloads and exposed through
`training.nonfinite_metrics=true` and `tracking/nonfinite_values`.

Resource sampling runs every 10 seconds, with an immediate initial sample.
Counters describe their named pod or process; they are not whole-cluster totals.
Storage I/O is block-device traffic, not occupied PVC capacity, and
`artifacts/output_directory_bytes` measures the output directory rather than the
job's incremental storage allocation. CPU time is cumulative seconds; memory and
GPU memory samples use bytes. These measurements are not billing or GPU-cost data.

### Current callback limits

The verified Training Hub interface exposes unified callbacks for `sft`, `osft`
and `lora_sft`. `embedding_sft` and the GRPO/verl profile do not currently expose
the same hooks. For profiles enabled in the API, common lifecycle/resource tracking
and available final numeric results still apply; this integration does not claim
per-step metrics for them or add an embedding execution profile. The entrypoint
detects support from the actual runtime signature, so future
profiles that expose `callbacks` use the same recorder automatically.

The unified interface has no before-save, save-failure, or checkpoint-restore
hooks. Consequently this integration cannot reliably measure checkpoint pause
time, identify every JIT save, prove restored optimizer state, or calculate work
lost on restart. It records backend-reported values and the observed initial
training step without inferring missing measurements. Some preemption paths exit
before emitting `on_save` or `on_train_end`.

Only the driver and callback-enabled trainer processes are instrumented. The
GPU-less GRPO head does not report node-wide GPU usage, and GRPO worker GPU
utilization is currently unavailable because those workers have no recorder hooks.
Missing or unrestricted device visibility also omits GPU samples. Per-rank
communication overlap and collective wait time require additional backend hooks.

## Platform setup

The default chart discovers the available `MLflow/mlflow` server and grants the
project's service-account group experiment access **in the installation project**.
This includes the service accounts generated by Ray for individual clusters.
It does not change the DataScienceCluster or replace an existing MLflow server.

If the platform does not have a server yet, an administrator enables the MLflow
operator and provisions the singleton following the
[MLflow setup guide](https://github.com/red-hat-data-services/red-hat-ai-examples/blob/main/examples/fine-tuning/mlflow.md).
For an initial development setup, the chart can provision a PVC-backed server:
set `mlflow.server.create: true` in the chart's `values.yaml` after enabling the
operator. A server created this way is retained when the chart is uninstalled.
Use the platform's production database and artifact storage guidance for a shared
or production installation.

For training in another project, grant its service-account group access once:

```bash
oc create rolebinding training-mlflow \
  --clusterrole=mlflow-operator-mlflow-integration \
  --group=system:serviceaccounts:YOUR_PROJECT \
  --namespace YOUR_PROJECT
```

Use the integration ClusterRole name installed by your MLflow operator. The
project must be exposed as an MLflow workspace under the server's workspace
selector. The submitting user also needs experiment access in that project.
Submission checks the caller and training account group; permission or readiness failures return
an actionable error before a RayJob is created.

The service reuses its configured platform gateway trust. Runtime authentication
uses the pod's mounted service-account token, reread for each request to support
rotation. Caller credentials are never embedded in the training program or run
metadata. No additional MLflow Python package is needed for the callback recorder.

Set `mlflow.enabled: false` in the chart to disable automatic tracking. If the
server is absent, requests without an experiment name can continue without
tracking. Once a server exists, readiness and authorization failures are errors;
they do not silently disable tracking. Transient delivery failures during an
accepted training run are reported without crashing training, with bounded queues
and flush timeouts. A forced process/node loss without API deletion can leave a
run open; there is no background history controller in this change.

Deletion through this API preserves metrics and finalizes an active or queued run
as `KILLED`, including when the driver cannot receive SIGTERM. Driver completion
and cancellation claim one immutable `training.terminal_result` parameter; retries
use that first result and its end time, preserving an earlier completion.

For a verified tracked job, DELETE adds the
`training.rhoai.io/mlflow-finalization` RayJob finalizer before requesting foreground
deletion. Callers need RayJob `patch` permission as well as `get` and `delete`.
The finalizer retains the UID-to-run binding until MLflow finalization succeeds.
If MLflow finalization fails, DELETE returns `503 metrics_unavailable`; restore
MLflow access and retry DELETE on the same job. The retained RayJob makes that
retry possible even after its workload stops. A successful retry preserves other
Kubernetes finalizers and releases only the tracking finalizer. These retries are
caller initiated and require the same project permissions; no credentials are
persisted for background processing. If the original deletion was rejected,
tracking is not finalized.

An uncertain submission response also retains its reserved run, even if the first
lookup returns 404: the original create may still commit after the timeout.
Check the job by name before resubmitting after a transport failure; the API does
not replay a possibly accepted submission. Reserved runs are removed only when
creation was not attempted or Kubernetes definitively rejected the create.

Tracking links are returned only when the configured server and the immutable run
binding match the current RayJob UID. Unverifiable links are omitted while ordinary
job status remains available. Jobs submitted before UID binding was introduced do
not gain tracking links automatically.

## Verify a deployment

Run the usual live training lifecycle with `--mlflow-url` pointing to the
administrator-provided managed server URL. For callback-enabled
algorithms this check verifies actual loss/step history and its persistence after
job deletion. GRPO verifies the common lifecycle and delivery health without
requiring unsupported per-step callbacks. This also applies to `--algorithm-matrix`:

```bash
make test-e2e E2E_ARGS="--job-file /path/to/request.json --queue-profile with-kueue --observe-cluster --check-logs --mlflow-url https://YOUR_GATEWAY/mlflow -v"
```

Use the service URL, project and public CA environment variables described in the
[E2E guide](../tests/e2e/README.md).
