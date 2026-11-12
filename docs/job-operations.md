# Training job operations

All paths below are relative to `/api/v1/projects/{project}/training-jobs`.
The project must already exist. An OpenShift bearer token is required;
TokenReview and project SubjectAccessReview run before backend access.
Kubernetes clients are request-local and use that caller's token, never the
service account's workload privileges. The service does not retain job state.

## Listing and details

`GET /` returns compact summaries: `id`, `status`, `created_at`, and
`queue_name` when present. It includes external caller-readable RayJobs, not
only API submissions. No `status` filter means all statuses. `status` and
`queue_name` are optional exact filters. `limit` defaults to 20.
`total_count` counts all matching jobs before API pagination, including all
Kubernetes inventory pages. `next_page_token` binds the project and filters;
each page rechecks permissions. Results are ordered by name, not a durable
snapshot. To distinguish a missing project from an empty inventory, listing
also requires caller access to read the target namespace.

`GET /{job_id}` returns available details from the RayJob, even after its
RayCluster has been cleaned up. The job ID is the namespaced RayJob name.
Optional details include timestamps, failure reason, a representable node
resource configuration, original submitter, and the accepted POST payload.
Submission annotations are validated against the contract before returning
them. Omitted settings stay omitted; external jobs without this metadata
remain readable. Raw runtime environments and Kubernetes Secrets are not
returned. The original submitter is informational, never an access-control
claim. Metrics/progress are omitted unless an authorized source provides them.
Resource summaries count the Ray head and active workers when their requested
CPU, memory, and GPU resources are homogeneous. Heterogeneous configurations
are omitted instead of being represented as an inaccurate per-node shape.

## Driver logs

`GET /{job_id}/logs?tail_lines=100` returns a `text/plain` driver-log snapshot.
`follow=true` returns transient `text/event-stream` events:

```text
event: log
data: {"message":"Epoch 1 complete\n"}

```

Following uses Ray's `tail_job_logs`; `tail_lines` applies only to snapshots.
Following checks backend status before sending HTTP 200 without first
downloading the full log. Later failures produce a
sanitized SSE `error` event. A stream lasts at most the deployment's configured
maximum (five minutes by default). Add `timeout_seconds=60` to request a shorter
connection; larger values are capped at that maximum. The parameter must use
ASCII digits without signs, leading zeros, whitespace, or a decimal point, in
the range 1 to 2147483647. Invalid values return 400 for both follow and snapshot
requests; valid values have no effect on snapshots. Expiry emits `stream_expired`
and closes the connection. Reconnecting revalidates authentication and
authorization. Disconnection closes the SDK iterator. Logs are not persisted
or aggregated across workers.

Before Ray submission, logs return `409 logs_not_ready`; KubeRay may already
have allocated a job ID at this point. If Ray reports that the submission is
absent, logs return `404 not_found`, not an instruction to keep retrying.
A removed RayCluster returns `410 ray_cluster_gone`. Gateway/version endpoint
failures remain `503 logs_unavailable`. These errors are returned as JSON before
opening an SSE stream.

The service automatically reads the ready platform
`GatewayConfig/default-gateway` and uses its published `status.domain` for
backend log access at `/ray/{project}/{rayClusterName}`. The chart grants the
service account only `get` access to this named configuration; job operations
still use the caller's identity. No backend URL is needed from dashboard users.

Gateway discovery runs on the first log request and caches the validated
address for 60 seconds. The default chart also reads the public ingress CA from
`openshift-config-managed/default-ingress-cert` using namespace-scoped, named
ConfigMap read permission. Customers do not need to create a CA ConfigMap.
The next log request after expiry refreshes platform metadata and certificates;
existing streams keep their original address and TLS context. Address changes
and their configuration source are logged. Failed refreshes log a warning and
keep the last validated gateway and CA available, retrying after five seconds.
Concurrent connections use that cache while one refresh is in progress. Without
a validated configuration, discovery failures return `503 logs_unavailable`
and retry after the same delay. Invalid local trust files still return `503`;
TLS certificate and hostname verification remain enabled. Job operations and
`/readyz` do not depend on logging availability.
Offline Helm rendering requires no cluster lookup.

For custom platforms, set an explicit HTTPS origin to bypass discovery:

```yaml
logBackend:
  gatewayURL: https://ray-gateway.example.com
```

The legacy `TRAINING_SERVICE_RAY_GATEWAY_URL` environment override remains
supported; configure it or the structured value once. Invalid overrides fail
validation without falling back to a discovered endpoint.

TLS verification is always enabled. Default platform trust combines normal public
roots and the discovered ingress CA. Optional overrides in the chart's
`values.yaml` select a custom gateway, an existing namespace-local CA ConfigMap,
or a public PEM bundle managed by Helm. `trustedCA.autoDiscover: false` disables
ingress CA discovery. Explicit gateways use normal container trust unless a
custom CA is supplied. `TRAINING_SERVICE_RAY_CA_FILE` remains supported for
non-Helm deployments. Do not set
`RAY_ADDRESS` or `RAY_API_SERVER_ADDRESS`, which override SDK address resolution.
No caller-supplied dashboard URL is used. RayJob and RayCluster reads use the
caller token and require a verified RayJob controller owner reference. Shared
or unowned clusters are not used as log endpoints. HTTP requests have bounded
timeouts and do not follow redirects with credentials, including WebSocket
handshakes. The pinned Ray SDK's streaming transport is overridden narrowly
to enforce that rule while retaining its verified TLS context.

## Deletion and GPU cleanup

`DELETE /{job_id}` returns **204 with no body** after Kubernetes accepts
foreground cascading deletion. This deletes the RayJob resource instance,
not the shared RayJob CRD. Kubernetes/KubeRay remove the owned RayCluster,
head/worker pods, and other owned dependents. GPU capacity becomes available
after the GPU pods terminate. The response does not mean GPU release has
already completed: finalizers, controllers, and pod termination are asynchronous.
Stalled finalizers must be investigated; the API does not force-remove them.

Pending, running, and terminal jobs can be deleted, regardless of execution
or deployment state. Deleting an active job cancels execution as Kubernetes
removes its owned cluster and pods. Shared or unowned clusters return **409**
without cleanup. UID and resource-version preconditions reject a concurrent
change with **409** rather than deleting a different or restarted job. Poll
`GET /{job_id}` until it returns **404** to confirm cleanup has finished.
An already-removed cluster does not prevent deletion of its job.
Existing input/output PVCs, checkpoints, model outputs, and referenced
Secrets are not explicitly deleted. Normal Kubernetes ownership rules still
apply to anything a job owner has configured as a dependent.

The caller needs permission to get/delete the RayJob and get its RayCluster
when present. No direct pod deletion or new service-account workload
privileges are introduced.

## Errors and verification

Submission and job operations share caller-scoped Kubernetes configuration.
`TRAINING_SERVICE_KUBERNETES_API_URL` must be a valid HTTPS URL without embedded
credentials, query parameters, or fragments. Invalid configuration fails closed
with `503 backend_unavailable` before a workload client is created. Configure
`TRAINING_SERVICE_KUBERNETES_CA_FILE` for a mounted private CA; TLS verification
is never disabled. Backend-originated `401` responses include
`WWW-Authenticate: Bearer`, just like authentication failures.

- 400: malformed path, query, or continuation token.
- 401/403: authentication or caller permission denied.
- 404: requested job/project or Ray submission absent.
- 409: shared/unowned cluster, concurrent deletion conflict, or driver logs not ready.
- 410: driver logs gone with the managed RayCluster.
- 503: authentication, authorization, Kubernetes, or Ray backend unavailable.

Errors use the contract's `code`/`message` envelope. Raw SDK exception bodies
are not sent to callers. HTTP status codes and symbolic error codes remain
separate: HTTP `404` uses `code: "not_found"`, not a numeric code in the body.
Known missing resources are described using their requested name and project:

```json
{
  "code": "not_found",
  "message": "Training job 'my-job' was not found in project 'team-alpha'."
}
```

Job reads, deletes (including a concurrent disappearance), and initial log
lookups use that same message. Missing project lookups identify the project;
submission lookups identify the missing PersistentVolumeClaim or Secret and
its project. Duplicate submissions identify the job and project. A missing
Ray submission is described separately from a missing job resource, with
job/project context, rather than being mislabeled as a startup delay.
Messages never echo backend response bodies, storage contents, or credentials.
An inaccessible resource still returns `403`; no extra privileged lookup is
made to infer whether it exists. Conflicts instruct clients to refresh before
retrying, and backend authentication failures preserve the Bearer challenge.

Run `make check`; focused unit tests mock Kubernetes and Ray and require no
cluster. Live E2E coverage is maintained separately from this change.

## Submission integration

Submission is provided by PR #7 and now shares
`api_impl/training_jobs.py` with list/detail/delete. The generated API group
selects one handler instead of dispatching individual operations between
competing subclasses. `observability.py` owns the logs route.

The job handler uses the `JobsBackend` interface in `jobs.py`; the logs response
uses the backend's `LogClient` interface. Backend API group/resource settings
also determine job authorization. The current adapter remains CodeFlare/Ray;
this separation does not claim Trainer support yet.

CodeFlare's `RayJobClient` provides log snapshot/status/tail operations.
Its underlying Ray transport is hardened to preserve existing request
timeouts, TLS verification, credential isolation and redirect protection.
See `logs.py` and the loopback transport regression tests.

The templates preserve submission's 202 status, Location header and request
context extension. Query failures remain 400; malformed JSON is 400 and
invalid POST bodies are 422. Submission and read/delete errors retain the
same `ErrorResponse` schema.

KubeRay ownership reference:
[RayJob controller](https://github.com/ray-project/kuberay/blob/master/ray-operator/controllers/ray/rayjob_controller.go)
sets a RayJob controller owner reference when creating a managed RayCluster.
The Ray SDK's
[job API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/doc/ray.job_submission.JobSubmissionClient.html)
provides log snapshots and following; deleting only an SDK record would not
release a Kubernetes RayCluster, which is why this endpoint deletes the CR.
