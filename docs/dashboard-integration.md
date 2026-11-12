# Dashboard API quickstart

Get the Training Service URL from your platform administrator. All endpoints
below start with `/api/v1`. Send the signed-in user's OpenShift bearer token on
every request; project permissions apply. Your dashboard does not need backend
URLs, backend SDKs, or Kubernetes job manifests.

For browser integration, use the dashboard's same-origin API proxy and forward
the user's bearer token. The service does not enable cross-origin browser access.
Keep tokens out of URLs and logs.

## Discover and submit

```bash
export SERVICE_URL=https://training-service.apps.example.com
export PROJECT=my-project
export TOKEN=$(oc whoami -t)
export JOB=my-training-job

curl --fail-with-body -H "Authorization: Bearer $TOKEN" "$SERVICE_URL/api/v1/algorithms"
curl --fail-with-body -H "Authorization: Bearer $TOKEN" "$SERVICE_URL/api/v1/algorithms/lora_sft"
```

Use discovery's parameter types, defaults, and required fields to build your form;
POST validates against that same catalog. Discovery can list more algorithms than
POST accepts; offer only `sft`, `osft`, `lora_sft`, and `lora_grpo` for submission. Prepare model/data storage and any
connection Secrets in the target project. Example `job.json`:

```json
{
  "name": "my-training-job",
  "model": {"uri": "pvc://training-data/models/my-model/"},
  "dataset": {"uri": "pvc://training-data/datasets/train.jsonl"},
  "algorithm": {"name": "lora_sft", "parameters": {"num_epochs": 1}},
  "resources": {"num_nodes": 1, "gpus_per_node": 1, "memory_per_node": "32Gi"},
  "output": {"destination_uri": "pvc://training-data/outputs/my-training-job/"}
}
```

Every submission requires `output.destination_uri`, using an existing PVC or S3
connection. Saved artifacts remain after the training cluster is cleaned up.

Use a unique job name. The project PVC and its files must already exist and be
accessible to the caller. Size `resources` to fit one GPU node; a job that cannot
be scheduled stays `PENDING` with no status message; delete it and resubmit
with smaller resources. HF and S3 connection-backed inputs are also supported;
see the [request examples](api/openapi.yaml).

```bash
export JOBS="$SERVICE_URL/api/v1/projects/$PROJECT/training-jobs"
curl --fail-with-body -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d @job.json "$JOBS"
curl --fail-with-body -H "Authorization: Bearer $TOKEN" "$JOBS/$JOB"
```

POST returns `202` with the job ID and a project-scoped `Location`. Poll GET for
`PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, or `STOPPED`. A `FAILED` job may
not include a reason; read its driver logs before deleting it. List jobs with GET
`$JOBS`; optional filters are `status`, `queue_name`, `limit`, and `page_token`.

## Logs and deletion

```bash
# Snapshot; tail_lines applies only to snapshots.
curl --fail-with-body -H "Authorization: Bearer $TOKEN" "$JOBS/$JOB/logs?tail_lines=100"
# Follow; timeout_seconds is optional and defaults to five minutes.
curl --no-buffer --fail-with-body -H "Authorization: Bearer $TOKEN" \
  "$JOBS/$JOB/logs?follow=true&timeout_seconds=300"
# Delete pending, running, or completed jobs. Active training stops.
curl --fail-with-body -X DELETE -H "Authorization: Bearer $TOKEN" "$JOBS/$JOB"
```

Following returns Server-Sent Events: `event: log` contains JSON with `message`;
`event: error` contains `code` and `message`. On `stream_expired`, reconnect with
a current token. If omitted, the timeout uses the administrator's default
(300 seconds initially); a smaller `timeout_seconds` shortens one connection,
and larger values are capped at that default.
Ingress timeouts can close a connection earlier. Disconnection does not stop
training. Logs are transient driver logs and can disappear after job cleanup.

Use a streaming `fetch` client that supports the Authorization header; native
browser `EventSource` cannot set it. Parse SSE events across response chunks,
and cancel the fetch when leaving the log view. A 200 means the stream opened;
handle subsequent SSE errors as well as HTTP errors.

DELETE returns `204` when cleanup starts. Poll GET until `404` before showing
cleanup complete. Stored inputs and outputs are preserved. A `409 conflict`
means refresh and retry; shared or unverified ownership is also rejected with
409 and must not be retried blindly.

Errors use `{"code":"...","message":"..."}`. Reauthenticate on 401, show an
access error on 403, show validation guidance on 400/422, and retry 409
`logs_not_ready` while a job starts. A 410 means logs are gone; 503 indicates an
unavailable or misconfigured service/backend. Queue discovery currently returns
501.
