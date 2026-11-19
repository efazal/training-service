# Kueue discovery and queued submission

GET queues and POST with an explicit or discovered default queue share a fail-closed preflight.
Authentication and project authorization run first. Namespace and LocalQueue reads
use the caller credential. Platform readiness reads use the service account and
only the named installation resources configured in `kueuePreflight` Helm values.
There are no DSC reads or dependencies, so an independently installed RHBoK is supported.

## Contract change

* 200: readiness and project eligibility passed; `kueue_enabled: true`.
  An eligible project with no queues returns `queues: []`.
* 409 `kueue_not_installed`: successful API discovery did not advertise Kueue.
* 409 `kueue_project_excluded`: effective configured selectors exclude the project.
* 404 `not_found`: caller project or explicitly selected LocalQueue is missing.
* 422 `validation_error`: a selected `queue_name` is not a DNS subdomain of at
  most 63 characters; rejected before any project or queue lookup.
* 400 `bad_request`: the project is not a valid OpenShift project name.
* 401/403: caller authentication/permission failure, including downstream failures.
* 503 `backend_unavailable`: incomplete/unavailable capability checks, unhealthy
  controller, unsupported configuration, a LocalQueue lookup answered with an unexpected
  status, or platform
  service-account permission failure.

This replaces the old absence response (`200`, `kueue_enabled: false`).
POST repeats preflight without requiring a preceding GET. Creation acknowledges
only the RayJob resource: admission remains asynchronous and may wait for quota,
GPUs or admission checks. The preflight is not an atomic guarantee: configuration
can change after any check; Kueue admission remains authoritative.

## Supported configuration and installation

Set the namespace, Deployment, ConfigMap, mutating webhook configuration, and
webhook Service names in the chart to match the actual installation. The chart
creates named-resource read-only permissions in that namespace, plus get on the
named webhook and Kueue discovery URLs. Callers need no platform/admin reads.


Readiness requires a served LocalQueue API, at least one available controller
replica, the configured RayJob integration, and a matching CREATE admission webhook
with a ready backend. The manager configuration's explicit namespace selector and
the webhook namespace selector both apply. Caller permission failures remain
401/403; unavailable or unrecognized capability checks fail closed with 503.

Readiness checks the registered webhook and a ready backend for its configured
Service port. It does not establish API-server network reachability or validate
`caBundle`/serving-certificate trust. Network policy or certificate rotation can
still cause submission to fail after successful preflight; admission is authoritative.

An available controller/webhook can serve during a rolling update. The preflight
does not reproduce operator ConfigMap digests or depend on config mount/argument
layout. It reads configured namespace policy; Kueue admission remains authoritative,
and checks cannot make configuration changes atomic with submission. Nonempty
webhook object selectors and CEL conditions are unsupported and fail closed.
Empty selector forms are equivalent and accepted.

On installation/upgrade Helm looks up the configured Kueue namespace and creates
its external-namespace Role/RoleBinding only if that namespace exists. It never
creates an operator namespace. Cluster-scoped discovery grants remain available
so an absent installation can be identified. If Kueue is installed later, run
`helm upgrade` again to provision the namespace-scoped grants. Offline `helm template`
cannot discover external namespaces; use connected server-side rendering for a
complete manifest when Kueue is installed. `createRBAC: false` is for externally
managed grants, not for disabling Kueue or discovery.

## Pinned SDK verification

Inspected the Red Hat `codeflare_sdk-0.40.1-1-py3-none-any.whl` pinned in
`pyproject.toml` (SHA256 `2edca4173a6db7169e4708d26395982a4e91c2aa2bcdf58c8e24737959cde96d`).
The source path `codeflare_sdk/ray/rayjobs/rayjob.py`, `RayJob.submit()` calls
`_build_rayjob_cr()`. With a managed cluster and explicit `local_queue`, that
method directly sets `kueue.x-k8s.io/queue-name`. It does not call
`local_queue_exists`, validate the namespace selector, or inspect controller health.
It relies on mutation to suspend the job. Separate cluster/queue SDK helpers are
not evidence of checks on this path. Discovery uses Kubernetes directly, not SDK helpers.

## Default queues

When `queue_name` is omitted or null, the service discovers a served LocalQueue
API version and lists queues with caller credentials. A single queue annotated
`kueue.x-k8s.io/default-queue: "true"` is checked with the same readiness,
eligibility, and existence preflight and passed explicitly to CodeFlare.
Confirmed Kueue absence or a complete list with no default permits unqueued
submission. Permission errors remain 401/403; ambiguous defaults, incomplete
responses, and unavailable discovery fail closed rather than silently dropping
the queue. Selected queue names must fit the RayJob label's 63-character limit.

CodeFlare 0.40.1's fallback lookup hardcodes `v1beta1` and suppresses 403/404.
For a confirmed unqueued submission, the request-scoped SDK transport reuses
our empty default result instead of repeating that lookup. No global SDK state
is patched. Configuration may still change after preflight.
