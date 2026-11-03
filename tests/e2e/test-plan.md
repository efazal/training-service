# Training Service — customer-readiness E2E test plan

Status: planned coverage, not an execution report. Last updated: 2 November 2026.

Implementation note (3 November): the follow-up request adds opt-in
Podman/Quay/Helm automation, superseding the original existing-deployment-only
boundary below. Cluster/operator/DSC/queue/storage provisioning remains excluded.
See [instructions](README.md) and [coverage status](coverage.md). IDs/priorities
below are the maintained backlog, not a list of passing tests.

Implementation tracking: [RHOAIENG-102547](https://redhat.atlassian.net/browse/RHOAIENG-102547).

## Purpose and boundaries

Build confidence in the live API before customer handoff, especially validation,
actionable errors, real training, security, and cleanup. Implement this plan through
small, focused PRs rather than bundling every test with a feature change.

- Provide an automated pytest suite that a developer/customer manually invokes
  after deploying Training Service with Helm. Replace repeated Swagger/curl
  walkthroughs with one documented local command against that existing service.
  Jenkins, scheduled CI, cluster provisioning, and automatic Helm installation
  are not deliverables of this test work.
- Exercise a deployed Training Service through HTTP, not FastAPI TestClient or a
  mocked backend. Keep implementation-specific edge cases in focused unit tests.
- Use the Ray-backed, standalone Helm Developer Preview as the initial baseline.
  Do not introduce Trainer, an operator, cancel/pause/resume, or new storage
  integrations merely to satisfy this plan.
- Test each advertised capability before customer use. A feature missing from an
  intermediate PR build may be explicitly skipped there, but a required feature
  skipped or failing in the customer build is a release gap, not a passing gate.
- This is a future verification backlog. It does not expand ticket scope, change
  the API contract, implement tests, or certify the current deployment.

## Contract and existing coverage

The release candidate's reviewed `training-service/docs/api/openapi.yaml`, packaged
OpenAPI, and published supported-configuration matrix are the test oracle. Compare
them with the live `/openapi.json`; do not infer support from Swagger examples alone.
Use the repository contract rather than historical external/local reference copies.

`tests.md`, `implementation-notes.md`, and parts of `architectire-dp.md` contain
historical scope or implementation assumptions. They must not override the selected
release contract. In particular, do not inherit TrainJob, automatic PVC creation,
pause/resume, or SDK-record-only deletion assumptions from those notes.

Reuse the local `training-service-96353/tests/e2e/` pytest suite when implementing:
it already has live probes/Swagger, basic authentication, listing, invalid-request,
missing-job, optional project-isolation, and opt-in training-lifecycle checks.
These are starting points, not proof of full coverage or customer readiness. The
existing lifecycle test does not independently verify checkpoints or GPU cleanup.

## Priorities and execution profiles

| Priority | Meaning |
| --- | --- |
| P0 | Customer-handoff gate for the advertised workflow: correctness, useful errors, security, real training, durable outputs, and safe cleanup |
| P1 | Broader regression, resilience, and supported-configuration coverage; required before handing off any configuration it covers |
| TP | Conditional scope requiring an explicit TP/product decision; not automatically a DP gate |

| Profile | Purpose and safeguards |
| --- | --- |
| Smoke | Read-only API checks; negative mutations use unique, nonexistent test names and verify no job was created |
| Training | Opt-in compute allocation; small model, tiny compatible dataset, bounded runtime, unique output prefix |
| Training without Kueue | Existing Ray-enabled deployment with Kueue absent/disabled, or an explicitly permitted unqueued project; omit `queue_name` and verify successful training without bypassing policy |
| Training with Kueue | Existing Ray/Kueue-enabled deployment with the supported DSC settings, RayJob integration, project enrollment, LocalQueue/ClusterQueue and quota configured; verify admission and successful training |
| Security | Distinct real users/service accounts and two projects; restricted credentials, not a cluster-admin token for denial assertions |
| Infrastructure | Read-only cluster/storage observer verifies ownership, cleanup, GPU requests, and artifacts independently of API responses |
| Resilience | Explicit opt-in on a dedicated test deployment; restart, dependency outage, race, and concurrency scenarios |
| Matrix / TP | Explicitly selected algorithm, runtime, storage, accelerator, queue, or disconnected configuration |

## Record the exact build being tested

For every run capture: API commit/image digest; branch and included/excluded PRs;
chart version and relevant non-secret values; OpenAPI version/checksum; RHOAI,
OpenShift, KubeRay, Ray client/runtime, CodeFlare SDK, and Training Hub versions;
runtime image digest; hardware/topology; queue policy; fixture versions; and test
profile. Use the immutable image digest, not just a mutable image tag or directory
name. Never record tokens, kubeconfig contents, Secret values, or credentialed URIs.

Paths below are relative to `/api/v1`. Test the exact customer candidate again after
adding/removing PRs; a green result on a personal or integration build does not
certify a different build.

## 1. Error contract and request validation

All errors must follow the reviewed envelope: stable string `code` and useful
`message`, correct HTTP status/content type, and applicable headers. Framework
validation must not unexpectedly return a different raw `detail` shape. Messages
must identify the invalid field/configuration and a corrective action without
exposing credentials, stack traces, internal exception bodies, or other tenants.

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| ERR-01 | P0 | Parameterize errors across submission, discovery, list, detail, logs, and delete; validate response schema, code/status pairing, and safe messages |
| ERR-02 | P0 | Malformed JSON → `400`; schema-invalid JSON → `422`; unsupported media type → `415`, according to the reviewed contract |
| ERR-03 | P0 | Missing/null/wrong-type required fields, empty body, array instead of object, invalid names, unknown fields → contract-defined rejection naming the offending field; no job created |
| ERR-04 | P0 | Missing/invalid/expired token → `401`; authenticated permission denial → `403`; dependency failure must not masquerade as either |
| ERR-05 | P0 | Missing authorized job/project → `404`; duplicate name or unsafe terminal deletion → `409`; unavailable implementation → `501` only where documented |
| ERR-06 | P0 | Authentication/authorization/Kubernetes/Ray infrastructure outage → documented `503` rather than a successful empty response or a misleading validation error |
| ERR-07 | P0 | Include multiple invalid parameters and nested fields; errors are deterministic, identify offending keys/paths and constraints, and do not merely say “Unsupported parameter” |
| ERR-08 | P1 | Invalid URL/path/query, unsupported HTTP method, empty query values, duplicate query keys, invalid enum/casing/encoding → agreed contract/routing behavior; no unintended fallback or mutation |
| ERR-09 | P1 | Unexpected internal failure → sanitized server-error response, not leaked traceback or falsely reported `422`; reproduce through safe test-environment fault injection |
| ERR-10 | P1 | Swagger examples, documented curl requests, and actual response media types/status codes agree; error guidance points only to documentation/endpoints available in that build |

### Regression: the reported `lora_sft` rejection

On 2 November the integration build accepted only `learning_rate`, `num_epochs`,
`effective_batch_size`, and `max_seq_len`. A request additionally containing
`max_tokens_per_gpu`, `warmup_steps`, `accelerate_full_state_at_epoch`, and
`checkpoint_at_epoch` received an opaque `422` before Ray submission.

Retain that parameter combination as a credential-free fixture. Against a runtime
catalog that supports it, it must be accepted and executed with those settings.
Against a deliberately restricted catalog, it must fail clearly, naming the
unsupported keys and the supported configuration. Do not permanently encode the
four-parameter restriction as the desired API behavior.

An illustrative improved response for the restricted build, using the existing
envelope rather than inventing a new `details` field:

```json
{
  "code": "validation_error",
  "message": "Unsupported algorithm.parameters for lora_sft: accelerate_full_state_at_epoch, checkpoint_at_epoch, max_tokens_per_gpu, warmup_steps. This deployment accepts: effective_batch_size, learning_rate, max_seq_len, num_epochs. Remove unsupported fields or use a runtime/catalog supporting them."
}
```

Treat this as acceptance intent, not an exact message snapshot or an implemented
fix. Discovery must expose the same supported parameters once implemented. Extra
structured error fields require a separate reviewed contract change.

## 2. Algorithm discovery and parameter validation

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| ALG-01 | P0 | Discover advertised algorithms; every entry corresponds to the configured runtime and tested Training Hub version, not an unrelated global registry |
| ALG-02 | P0 | List response is understandable and follows the agreed compact/detail contract; no accidental raw Python docstring dump; schemas remain available through the approved interface |
| ALG-03 | P0 | Unknown algorithm and unsupported parameter → `422` identifying the name/key and how to discover supported choices; no backend job created |
| ALG-04 | P0 | Parameterize each supported algorithm's required/optional fields, defaults, enum values, ranges, and nullable behavior against its runtime schema |
| ALG-05 | P0 | Valid zero values such as `warmup_steps: 0` and boolean `false` survive validation and launcher mapping; booleans must not be mistaken for integers |
| ALG-06 | P0 | Incorrect number/string/boolean/array/object types, negative/out-of-range values, missing algorithm-required fields, and unsupported combinations are rejected at the appropriate boundary with an actionable explanation |
| ALG-07 | P0 | Accepted parameters reach Training Hub unchanged or through documented normalization; they are not silently dropped, renamed, or replaced with hard-coded defaults |
| ALG-08 | P0 | User attempts to override service-managed topology, paths, credentials, or launcher behavior through algorithm parameters are rejected or handled exactly as the reviewed schema permits |
| ALG-09 | P1 | Change runtime/catalog version in an isolated deployment; discovery and validation remain in sync; missing/corrupt/stale catalog is diagnosed, not advertised as an empty healthy catalog |
| ALG-10 | P1 | Registry membership alone is insufficient: cover every advertised algorithm/backend/topology, including algorithms whose Python APIs require callbacks or extra services; unsupported REST configurations are not advertised |

## 3. Submission and actual training

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| JOB-01 | P0 | Small known-good request → `202`, valid job ID and `Location`; accepted job becomes queryable in the supplied existing project; `202` does not mean training succeeded |
| JOB-02 | P0 | Follow a real job to `SUCCEEDED`, inspect meaningful progress, and verify non-empty, usable model/checkpoint artifacts using a format-appropriate check; reaching any terminal state is not enough |
| JOB-03 | P0 | Missing project never defaults to the service namespace or creates a namespace; an existing empty project is distinguishable from an absent/forbidden project |
| JOB-04 | P0 | Rejected requests create no RayJob, RayCluster, pods, or storage artifacts; use a bounded observer window and exact test-owned identifiers |
| JOB-05 | P0 | Duplicate name → `409` without duplicate execution; the same name in a different authorized project is independent |
| JOB-06 | P0 | Named model, dataset, parameter, resource, and output overrides are honored; GET details preserve the accepted POST configuration and authenticated submitter without secrets |
| JOB-07 | P0 | Controlled runtime failure → API-visible `FAILED`, terminal timestamp and safe actionable failure information; never `SUCCEEDED` or an indefinitely stale running state |
| JOB-08 | P0 | Dataset/model incompatibility, malformed/empty data, missing columns/files, unreadable paths, and inaccessible remote assets produce clear validation or asynchronous failure diagnostics at the boundary capable of detecting them |
| JOB-09 | P1 | Omitted optional fields, explicit defaults, and allowed null/empty values follow the documented semantics; omitted output behavior is explicit and never falsely promises durable checkpoints |
| JOB-10 | P1 | Alternate small models/datasets, supported instruction/chat formats and field mapping work without a fixture-specific implementation |
| JOB-11 | P1 | Single-/multi-GPU and multi-worker cases use the documented head/worker topology, resources, and storage access mode; no silently ignored distributed settings |
| JOB-12 | P1 | Requested known but unimplemented backend gets the documented `501`; invalid backend enum/type gets schema validation, not silent Ray fallback |

Known separate risk: earlier live LoRA execution failed inside the selected
Training Hub/Unsloth runtime with an `assistant_only_loss`/dataset-format error.
Keep a regression workload for this, independently of the request-validation fix.
Record the failing image digest and prove the corrected runtime succeeds; changing
API validation alone must not be reported as resolving training execution.

## 4. Authentication, authorization, and tenant isolation

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| SEC-01 | P0 | Human and service-account tokens authenticate as their actual identity; missing, malformed, revoked, invalid, and expired tokens fail safely |
| SEC-02 | P0 | Use real create/read/delete/log-restricted personas; test every relevant endpoint and paginated request against the permitted and forbidden project |
| SEC-03 | P0 | Project membership alone does not imply all resource permissions; caller lacking job create/delete, required Secret/PVC access, or log permissions is denied without service-account fallback |
| SEC-04 | P0 | Same-name jobs in separate projects remain isolated; forged submitter metadata and impersonation headers cannot grant access or change recorded caller identity |
| SEC-05 | P0 | Concurrent submissions/reads/log requests by different identities never reuse another caller's credentials, Kubernetes client, cached response, or backend result |
| SEC-06 | P0 | API responses, its own diagnostics, persisted submission metadata, generated launch configuration, and test reports contain no caller bearer tokens or resolved Secret values |
| SEC-07 | P0 | TLS verification stays enabled for API/Kubernetes/Ray/storage paths; trusted custom CA works; untrusted certificates fail without an insecure fallback |
| SEC-08 | P1 | Permission removal between list pages or reconnecting logs is enforced; token expiry does not switch to a privileged identity or stop an already accepted training workload |
| SEC-09 | P1 | Caller-controlled project/job/URI/continuation-token input cannot cross tenant boundaries, inject launcher commands, escape PVC mounts, or redirect bearer credentials to another host |
| SEC-10 | P1 | ServiceAccount workload privileges remain least-privilege; test-role setup does not accidentally grant a denied user extra access; authorization precedes sensitive existence disclosure |

## 5. Model, dataset, credentials, and output storage

Cover only the approved storage matrix. References use the current URI contract:
`hf://<secret-ref>/<repository>`, `pvc://<pvc>/<path>`, and
`s3://<connection-or-secret-name>/<bucket>/<path>`, as defined for the candidate.
Do not reintroduce removed `source`, `split`, or separate `secret_ref` fields.

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| DATA-01 | P0 | Supported local PVC model/dataset inputs and durable PVC outputs work end to end; missing/unbound/forbidden PVC, file, and unwritable destination are explained |
| DATA-02 | P0 | Supported public/private/gated Hugging Face inputs resolve using the referenced project-local credential; missing/invalid credential and denied model access are distinguishable without leaking values |
| DATA-03 | P0 | Supported S3-compatible input/output uses the selected bucket, endpoint, prefix, credential, and trusted CA; prove upload completes and output remains readable |
| DATA-04 | P0 | Missing/forbidden/malformed Secret or connection, missing bucket/key, bad endpoint/TLS, expired storage credential, and denied read/write produce useful errors or terminal failure reasons |
| DATA-05 | P0 | Unsupported URI scheme, malformed/missing URI components, embedded credentials, traversal (including encoded variants), and cross-project references fail safely without unintended reads/writes |
| DATA-06 | P0 | Terminal job deletion preserves customer-provided PVCs, Secrets, dataset/model files, and saved outputs; each test uses its own output prefix |
| DATA-07 | P1 | RWO/RWX and multi-node scheduling follow the supported storage requirements; incompatible storage/topology never appears as unexplained successful submission followed by silent data loss |
| DATA-08 | P1 | Runtime disk full, quota exhaustion, interrupted download/upload, and output collision have clear failure behavior; partial outputs are not advertised as a usable completed model |
| DATA-09 | P1 | Native Azure Blob/GCS/datalake URIs are either validated supported integrations or clearly unsupported; S3 compatibility is not a promise of every provider's native API |

## 6. Resources, queues, and scheduling

Run the training lifecycle in **both** configurations, not just a mocked queue
validation test:

- **Without Kueue:** use an existing environment where Kueue is absent/disabled,
  or where policy explicitly permits an unqueued job. Omit `queue_name` entirely
  (an empty string is not the no-queue fixture). Verify `202`, actual execution,
  `SUCCEEDED`, outputs, and safe terminal cleanup without requiring queue APIs.
  Record whether Kueue was absent or an explicit opt-out was used; an opt-out
  does not prove the absent-Kueue case.
- **With Kueue:** prerequisites are Ray and Kueue enabled/configured through the
  supported DataScienceCluster (DSC) settings, applicable RayJob integration and
  namespace enrollment, an active project LocalQueue backed by a ClusterQueue,
  appropriate ResourceFlavor/quota, and caller permissions. Submit to the named
  LocalQueue; verify the queue is honored, the associated Workload is admitted,
  and training reaches `SUCCEEDED`, retains outputs, and cleans up safely.
- A dedicated opt-in fixture may hold/release test-owned quota to demonstrate
  pending-to-admitted behavior. Never manipulate shared queues or toggle the
  cluster's DSC from the default test command. Operators prepare each environment
  before invoking the suite; the harness records and validates that profile.
- In a project that requires admission, omitting `queue_name` must never bypass
  Kueue. Missing queues, disabled integrations, and quota exhaustion need distinct,
  actionable diagnostics according to the agreed contract.

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| QUEUE-01 | P0 | No queue supplied works only where cluster policy permits; never bypass mandatory Kueue admission or silently choose a different queue |
| QUEUE-02 | P0 | Authorized queue discovery and explicit queue selection agree; target queue is preserved in accepted job details and actual SDK submission |
| QUEUE-03 | P0 | Missing/invalid/forbidden queue, Kueue absent, disabled integration, and unavailable queue API follow documented distinctions; never fabricate a healthy empty list |
| QUEUE-04 | P0 | Queued job with exhausted quota remains visibly pending/queued, not running; admission eventually progresses when capacity is available |
| QUEUE-05 | P0 | Invalid CPU/memory quantities, unsupported GPU resource, non-positive topology, and incompatible resource/algorithm settings fail clearly according to the schema |
| QUEUE-06 | P1 | Valid fractional/millicore CPU and supported memory units map correctly; observed requests/limits match the documented head/worker resource mapping |
| QUEUE-07 | P1 | GPU scarcity, unschedulable placement, project quota, queue suspension, and workload-policy admission failures surface intelligible reasons without API hangs |
| QUEUE-08 | P1 | Verify every advertised RHOAI/KubeRay/CodeFlare/Ray client/runtime combination; do not claim compatibility solely because Helm installs |
| QUEUE-09 | P0 | Run a real no-queue lifecycle with Kueue absent/disabled; omit `queue_name`, require `SUCCEEDED` and durable artifacts, and independently verify test-owned resource/GPU cleanup |
| QUEUE-10 | P0 | Run a real queued lifecycle with Kueue enabled/configured in the DSC and the required RayJob/namespace integration; verify LocalQueue selection, Workload admission, `SUCCEEDED`, durable artifacts, and resource/GPU cleanup |
| QUEUE-11 | P0 | In a mandatory-admission project, missing queue selection cannot start an unadmitted workload; verify the documented rejection/pending behavior without silently opting out |

## 7. Listing, detail, timestamps, and pagination

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| READ-01 | P0 | List returns compact summaries and accurate `total_count`, not full submissions; each item has required ID/status/creation time |
| READ-02 | P0 | Without `status`, include all authorized statuses; supported status/queue filters are exact; count reflects all matches before pagination |
| READ-03 | P0 | Details include available accepted request, submitter, resources, data/model/output references, status and failure information; defaults/omissions are represented honestly |
| READ-04 | P0 | `created_at`, `started_at`, and `finished_at` follow the contract: no invented timestamps, valid timezone format, chronological ordering, terminal finish where available |
| READ-05 | P0 | Failed-before-start jobs and external SDK-created jobs remain readable; absent API annotations do not cause `500`, fabricated submitters, or leaked raw runtime environments |
| READ-06 | P1 | Empty and multi-page results, default/min/max limits, every Kubernetes inventory page, final-page token, and count/filter consistency are correct |
| READ-07 | P1 | Tampered/malformed token and reuse against another project/filter are rejected; changed inventories follow documented non-snapshot semantics without unauthorized exposure |
| READ-08 | P1 | Detail requests continue after API restart and RayCluster shutdown wherever retained RayJob metadata allows; unavailable telemetry is omitted or explained, not invented |

## 8. Driver logs and streaming

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| LOG-01 | P0 | Authorized snapshot returns non-empty driver logs with correct `text/plain` content type and documented `tail_lines` behavior |
| LOG-02 | P0 | Missing job, not-yet-ready logs, gone RayCluster/logs, forbidden caller, and backend outage produce the agreed `404`/`409`/`410`/`403`/`503`, not indistinguishable empty success |
| LOG-03 | P0 | Follow mode uses the documented SSE envelope/content type; actual new log lines arrive during training; no requirement for unsupported worker aggregation |
| LOG-04 | P0 | Pre-stream failure returns the proper HTTP error; failure after headers uses a sanitized SSE error event, not silent termination presented as success |
| LOG-05 | P1 | Invalid `follow`/`tail_lines`, Unicode/multiline/large logs, and empty-but-valid snapshots follow documented parsing/framing semantics |
| LOG-06 | P1 | Disconnect/reconnect, slow consumer, stream timeout, job completion, token/permission changes, and repeated streams leave no leaked SDK iterator or unbounded buffering |
| LOG-07 | P1 | Logs resolve only to a verified job-owned RayCluster through the trusted gateway; forged/unowned/shared cluster references and redirects cannot expose other workloads or credentials |

## 9. Terminal deletion, ownership, and GPU release

DELETE is terminal-job cleanup, not cancellation. It removes a RayJob resource
instance and its eligible owned resources, never the cluster-wide RayJob CRD.

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| DEL-01 | P0 | `SUCCEEDED`/`FAILED`/`STOPPED` eligible job → `204` with no body; bounded polling eventually returns job `404` and removes it from list |
| DEL-02 | P0 | Independently inspect owner UIDs: owned RayCluster/head/worker/submitter resources disappear and no test-owned GPU pods remain; `204` or API `404` alone is insufficient |
| DEL-03 | P0 | Compare test workload GPU requests and pod termination to prove allocations are released; global free-GPU count need not increase if other workloads consume capacity |
| DEL-04 | P0 | Running/pending/unknown/retrying job, shared or unowned cluster → `409` with explanation and no destructive side effects |
| DEL-05 | P0 | Saved outputs and pre-existing storage/credentials/unrelated workloads survive; the RayJob CRD remains installed; other API jobs remain functional |
| DEL-06 | P1 | Already-cleaned-up cluster and failed-before-cluster-start job can be cleaned safely; repeat DELETE follows documented `404`/idempotency behavior |
| DEL-07 | P1 | UID/resource-version change, deletion/recreation race, or ownership mismatch cannot delete a replacement job or unrelated cluster |
| DEL-08 | P1 | Finalizer/controller stall remains observable with bounded timeout and useful diagnostics; the API/test harness never force-removes finalizers to manufacture a pass |

## 10. Deployment, recovery, concurrency, and compatibility

| ID | Priority | Scenario and expected result |
| --- | --- | --- |
| OPS-01 | P0 | Against the exact candidate chart/image already installed in a dedicated namespace, verify readiness, liveness, Swagger, live OpenAPI, route TLS, auth configuration, and minimal real training; deployment is a documented prerequisite, not performed by the test runner |
| OPS-02 | P0 | Missing runtime/gateway/CA/auth configuration fails clearly at readiness or operation time; probes must not falsely certify an unusable training workflow |
| OPS-03 | P0 | API restart/upgrade leaves an accepted job running and inspectable, with accepted metadata and output location preserved; API state is not the only copy |
| OPS-04 | P0 | Lose a POST response after backend acceptance; reconcile by project/name before retry; at most one execution for that name and clear duplicate/recovery guidance |
| OPS-05 | P1 | Concurrent same-name POSTs create at most one job; distinct users/jobs/outputs remain independent; bounded parallel reads/submissions do not corrupt caller scope |
| OPS-06 | P1 | TokenReview/SAR/Kubernetes/Ray outage and recovery, request deadlines, SDK exceptions, and TLS/auth failures are mapped consistently and recover without restart-dependent stale credentials |
| OPS-07 | P1 | DP-to-candidate upgrade preserves running jobs and supported existing job reads/API contract; test the actual DP metadata, labels, and identifiers rather than newly created jobs only |
| OPS-08 | P1 | Separate Helm service namespace and target workload project work correctly; test non-admin client identities and supported ingress/gateway configurations |
| OPS-09 | P1 | Small bounded load/long-running stream run checks connection/resource leaks and responsiveness; numeric SLOs, body-size limits, `413`/`429`, and rate limiting require separately agreed contracts |

## 11. Conditional TP / future coverage

Do not turn these into new DP dependencies or silently claim they are supported.
If a capability is included in the TP customer promise, its corresponding checks
become release gates for that scope.

- Disconnected: mirror every referenced image; pre-stage model, tokenizer,
  dataset and packages; verify submission-to-usable-output with external egress
  blocked, internal credentials/CA trusted, and no hidden Hugging Face/PyPI or
  telemetry dependency. Cover install/upgrade and actionable missing-asset errors.
- ROCm/CPU: validate the actual Training Hub-enabled Ray runtime, algorithm and
  hardware combination. Generic Ray images or Trainer universal images alone are
  not evidence of support.
- Operator/RHOAI integration: test enable/disable, reconciliation, permissions,
  config changes, mirroring and upgrades only after architecture/ownership and
  preservation semantics are agreed.
- Native cloud/datalake providers, estimation, custom code, MLflow, additional
  backends, cancellation, checkpoint resume, and automatic recovery need their own
  approved scope, error contracts, security review, and tests.

## 12. Test implementation and safety rules

- Document one local entry point, required URL/project/token/CA configuration,
  supported markers/options, both Kueue profiles, fixture preparation, and rerun
  commands. Failures return a nonzero exit status and useful sanitized diagnostics;
  missing prerequisites are explicit, not a misleading all-skipped green run.
- Keep live tests opt-in and default unit-test commands unit-only. Reuse existing
  pytest/HTTP fixtures; avoid another test framework or duplicated API clients.
- Use dedicated projects, tiny fixtures, cost/runtime limits, unique short names,
  and per-run output directories. Record job/owner UIDs before cleanup.
- Supply tokens only via environment/approved credential handling; verify TLS.
  Never commit credentials, enable shell tracing, print HTTP auth/debug data, or
  include secrets in test parameters, snapshots, JUnit, or failure artifacts.
- An optional observer can inspect cluster state with separately declared
  read-only privileges; it must never replace the restricted HTTP caller or make
  a denied API request succeed.
- Use polling with explicit deadlines and state diagnostics, not fixed sleeps.
  Separate submission, scheduling, training, stream, and cleanup timeouts.
- Never automatically retry POST after ambiguous transport errors. Recover by
  the stable name and verify whether the first request was accepted.
- Finalizers run after failures, but cleanup targets only this run's verified
  resources. Do not delete active jobs through an unsupported cancel path, use
  broad namespace selectors, remove CRDs, or escalate privileges.
- Retained diagnostic jobs/outputs and interrupted runs are reported explicitly.
  Never delete customer storage; keep artifacts until verification is complete.
- Destructive faults, RBAC changes, deployment restarts, and queue manipulation
  require dedicated opt-in fixtures and exact restoration of test-owned state.
- No skip/xfail may conceal a known customer-blocking failure. Infrastructure
  absence, expected validation rejection, and product regression are distinct
  outcomes. `202`, terminal `FAILED`, and `204` are not substitutes for success,
  usable artifacts, and resource cleanup respectively.

## 13. Small, incremental PR batches

Keep each batch independently reviewable; add a feature's unit tests in the
feature PR and its live regression in the smallest relevant E2E PR. Do not wait
for a wholesale testing ticket or bundle unrelated product refactors.

| Batch | Focus | Initial cases |
| --- | --- | --- |
| 1 | Actionable validation/errors; reproduce the reported payload | ERR-01–07, ALG-03–08; server fix and focused regressions together |
| 2 | Real golden-path training and durable artifacts, with and without Kueue | JOB-01–08, DATA-01/06, QUEUE-09–11; runtime failure tracked independently |
| 3 | Caller permissions, service accounts, cross-project isolation | SEC-01–06, including concurrent caller isolation |
| 4 | List/detail/count/timestamp/pagination accuracy | READ-01–08 |
| 5 | Driver log snapshots, streams, and error paths | LOG-01–07 |
| 6 | Terminal cleanup, ownership, and independently observed GPU release | DEL-01–08 |
| 7 | Discovery/catalog correctness and supported algorithm/runtime matrix | Remaining ALG cases and supported JOB-10/11 combinations |
| 8 | Storage/credential and queue/resource matrices | Remaining DATA and QUEUE cases, split again by provider/policy as needed |
| 9 | Restarts, ambiguous retries, DP continuity, bounded concurrency | OPS cases, split by isolated fault scenario |
| Separate TP work | Disconnected, new accelerators, operator lifecycle | Only after scope/owners/runtime delivery are agreed |

Batch numbering is a review sequence, not permission to hand off while required
P0 coverage is incomplete. Attach the tests for each capability to its delivery.

## 14. Customer-handoff checklist and evidence

- [ ] Agree advertised capabilities and close contract decisions below.
- [ ] Re-run on the exact candidate build and customer-relevant supported matrix.
- [ ] All applicable P0 cases pass; every advertised additional configuration has
  its relevant P1/TP evidence; no unexplained skip or known blocking failure.
- [ ] A real job reaches `SUCCEEDED`; usable artifacts survive terminal cleanup.
- [ ] Both without-Kueue and with-Kueue training profiles pass on their prepared
  deployments; mandatory-admission policy is never bypassed.
- [ ] Restricted users/service accounts pass isolation and no-elevation checks.
- [ ] GPU/owned-resource cleanup is independently verified; no test leaks remain.
- [ ] Error messages identify unsupported fields and corrective action; docs and
  discovery accurately reflect the shipped runtime rather than future support.
- [ ] Record sanitized test/JUnit results, build matrix, job IDs, timestamps,
  artifact verification and cleanup evidence; link defects to failed case IDs.
- [ ] Publish known limitations and exact reproduction commands without tokens.

## Decisions to settle before automating affected cases

1. Discovery's compact/detail schema and authoritative runtime/catalog version
   binding; the source of parameter constraints and supported algorithm matrix.
2. Unsupported-feature, missing-project/path, absent-Kueue, storage-access, and
   late-runtime-error status/code semantics not already settled by OpenAPI.
3. Completed-job metadata/log retention after RayCluster shutdown; no accidental
   promise of durable logs without an implemented retention source.
4. Accepted request/default representation, optional output durability, and which
   artifact types constitute a usable model for each advertised algorithm.
5. Exact queue/no-queue policy, topology/storage constraints, and minimal caller
   and observer RBAC for every supported installation profile.
6. DP-to-TP compatibility guarantees and which TP-only capabilities are actually
   required for the target release.

References: [reported DP work](https://redhat.atlassian.net/browse/RHAISTRAT-2712),
[TP draft](https://redhat.atlassian.net/browse/RHAISTRAT-3069),
[Training Service repository](https://github.com/opendatahub-io/training-service).
