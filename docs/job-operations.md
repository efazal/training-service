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
Optional details include timestamps, failure reason, a representable worker
resource configuration, original submitter, and the accepted POST payload.
Submission annotations are validated against the contract before returning
them. Omitted settings stay omitted; external jobs without this metadata
remain readable. Raw runtime environments and Kubernetes Secrets are not
returned. The original submitter is informational, never an access-control
claim. Metrics/progress are omitted unless an authorized source provides them.

## Driver logs

`GET /{job_id}/logs?tail_lines=100` returns a `text/plain` driver-log snapshot.
`follow=true` returns transient `text/event-stream` events:

```text
event: log
data: {"message":"Epoch 1 complete\n"}

```

Following uses Ray's `tail_job_logs`; `tail_lines` applies only to snapshots.
The backend is checked before sending HTTP 200. Later failures produce a
sanitized SSE `error` event. A stream lasts at most five minutes; reconnecting
revalidates authentication and authorization. Disconnection closes the SDK
iterator. Logs are not persisted or aggregated across workers.

The administrator must configure a trusted RHOAI HTTPS gateway serving
authenticated Ray endpoints at `/ray/{project}/{rayClusterName}`. Use the
chart's existing `env` values, for example:

```yaml
env:
  - name: TRAINING_SERVICE_RAY_GATEWAY_URL
    value: https://ray-gateway.example.com
```

TLS verification is always enabled. The gateway certificate must be trusted
by the container. `TRAINING_SERVICE_RAY_CA_FILE` can select a CA bundle that
is already present in the image or mounted by the deployment. Do not set
`RAY_ADDRESS` or `RAY_API_SERVER_ADDRESS`, which override SDK address resolution.
No caller-supplied dashboard URL is used. RayJob and RayCluster reads use the
caller token and require a verified RayJob controller owner reference. Shared
or unowned clusters are not used as log endpoints. HTTP requests have bounded
timeouts and do not follow redirects with credentials.

## Deletion and GPU cleanup

`DELETE /{job_id}` returns **204 with no body** after Kubernetes accepts
foreground cascading deletion. This deletes the RayJob resource instance,
not the shared RayJob CRD. Kubernetes/KubeRay remove the owned RayCluster,
head/worker pods, and other owned dependents. GPU capacity becomes available
after the GPU pods terminate. The response does not mean GPU release has
already completed: finalizers, controllers, and pod termination are asynchronous.
Stalled finalizers must be investigated; the API does not force-remove them.

Only execution states `SUCCEEDED`, `FAILED`, and `STOPPED` are eligible.
Pending/running jobs, unknown execution states, retrying deployments, and
shared or unowned clusters return **409** without cleanup. This is not a
cancel endpoint. UID and resource-version preconditions reject a concurrent
change with 409 rather than deleting a different or restarted job.
Already-cleaned-up clusters do not prevent deletion of their terminal job.
Existing input/output PVCs, checkpoints, model outputs, and referenced
Secrets are not explicitly deleted. Normal Kubernetes ownership rules still
apply to anything a job owner has configured as a dependent.

The caller needs permission to get/delete the RayJob and get its RayCluster
when present. No direct pod deletion or new service-account workload
privileges are introduced.

## Errors and verification

- 400: malformed path, query, or continuation token.
- 401/403: authentication or caller permission denied.
- 404: requested job/project absent.
- 409: unsafe deletion or driver logs not ready.
- 410: driver logs gone with the managed RayCluster.
- 503: authentication, authorization, Kubernetes, or Ray backend unavailable.

Errors use the contract's `code`/`message` envelope. Raw SDK exception bodies
are not sent to callers. Run `make check`; focused unit tests mock Kubernetes
and Ray and require no cluster. Submission, discovery, and estimation are
not implemented by this ticket.

## Submission integration

The pending submission implementation remains a separate change. Generated
routes select explicit operation owners, not the first imported handler.
`api_impl/job_operations.py` owns list/detail/delete and `observability.py`
owns logs. An older submission handler can continue to own POST, even when
it contains unimplemented read-method stubs. Duplicate owners fail closed.
The templates preserve submission's 202 status, Location header, and request
context extension when generating from the combined contract. Query failures
remain 400; malformed JSON is 400 and invalid POST bodies are 422. Submission
and job-operation errors share the generated `ErrorResponse` schema.

KubeRay ownership reference:
[RayJob controller](https://github.com/ray-project/kuberay/blob/master/ray-operator/controllers/ray/rayjob_controller.go)
sets a RayJob controller owner reference when creating a managed RayCluster.
The Ray SDK's
[job API](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/doc/ray.job_submission.JobSubmissionClient.html)
provides log snapshots and following; deleting only an SDK record would not
release a Kubernetes RayCluster, which is why this endpoint deletes the CR.
