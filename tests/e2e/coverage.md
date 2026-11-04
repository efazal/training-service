# Coverage and customer-readiness gaps

Source: [RHOAIENG-102547](https://redhat.atlassian.net/browse/RHOAIENG-102547) and
[the full plan](test-plan.md). This PR is a repeatable baseline, **not a claim
that every P0 or configuration passes**. Mocked unit tests are not live evidence.

The follow-up requests add Podman → Quay → Helm automation, a fresh labeled
namespace, LocalQueue/PVC/fixture provisioning, and mandatory Kueue-admitted
training. They supersede the ticket/plan's original “already deployed only”
boundary. Shared cluster/operator/DSC/ClusterQueue/StorageClass provisioning and
Jenkins remain excluded; users enable Ray/Kueue in the DSC first.

| Plan IDs | Implemented test / remaining status |
| --- | --- |
| ERR-01–05/10 (partial) | `test_service.py`: envelope/media type, probes/OpenAPI, auth, malformed JSON/schema/media type, invalid queries/URI/resources/algorithm, missing job/project/PVC/Secret; duplicate during lifecycle |
| ERR-06–09 (partial/opt-in) | Prepared invalid Kubernetes URL regression; broader outages/faults and field-level guidance deferred |
| ALG-03/06/08 (partial) | Rejection matrix: unknown algorithm, bool-as-number, unsupported/invalid parameters, traversal; API confirms no job |
| ALG-01/02/04/05/07/09/10 (conditional/deferred) | Discovery is 501 on main. Dynamic catalog/mapping, zero/false acceptance and full algorithm matrix need corresponding features |
| Reported LoRA regression, ERR-07 | Original extra parameters preserved. Main's `Unsupported lora_sft parameter.` is a **known actionable-error gap**; envelope/rejection checks do not certify field guidance or parameter execution |
| JOB-01/02/06, DATA-01/06 (partial) | Lifecycle: 202/Location, duplicate, request/submitter/resources, real SUCCEEDED; default deploy run requires read-only LoRA artifact-format check before/after delete, not inference evaluation/all storage providers |
| JOB-03/04/12 (partial) | Missing project/resources, rejected-job lookup, KFT 501 vs invalid backend 422. Independent cluster/storage no-side-effect proof for every negative case deferred |
| JOB-05/07–11 (partial/deferred) | Same-project duplicate covered; cross-project names, controlled runtime failure, dataset matrix and multi-node/GPU follow-ups |
| SEC-01/06/07 (partial) | Missing/invalid/malformed token, Bearer challenge, token echo guard, verified TLS/no redirects/proxies. Expired/revoked personas, Secret scans and adversarial TLS deferred |
| SEC-02–05/08–10 (partial/deferred) | Optional forbidden-project list; complete real least-privilege endpoint/persona matrix, permission changes and concurrent isolation deferred |
| DATA-02–05/07–09 (partial/deferred) | Missing HF/S3 Secret/PVC context and PVC traversal rejection; positive inputs from chosen fixture. Private/gated/S3 outputs, credential/storage failures and access-mode matrix need dedicated fixtures |
| QUEUE-01/02/09/10 (profile-dependent) | Default deploy run requires DSC Ray/Kueue Managed+Ready, labeled fresh namespace, LocalQueue, owned admitted Workload, SUCCEEDED, artifacts and GPU cleanup. Existing-deployment absent-Kueue/unqueued-opt-out profiles remain separate |
| QUEUE-03–08/11 (partial/deferred) | Invalid resources/empty queue covered. Quota hold/release, mandatory-admission denial, disabled integrations and compatibility matrix require prepared scenarios |
| READ-01–04/06/07 (partial) | Compact summaries/count bounds, pagination, created-job inclusion/status filters, payload/optional truthful timestamps, malformed token; exact inventory count/default-status/filter binding matrix deferred |
| READ-05/08 (deferred) | External/failed-before-start records, metadata after API restart require prepared scenarios |
| LOG-01–05 (partial) | Snapshot and two distinct SSE events during training; missing job both modes, optional missing Ray submission, invalid queries, notready/gone/backend statuses. Post-header faults, tail/Unicode/large-log matrix deferred |
| LOG-06/07 (deferred) | Closing successful follow exercises disconnect; reconnect, slow consumer, lifetime/leaks and malicious ownership/gateway fixtures still required |
| DEL-01–03/05/06 (partial) | Terminal delete/eventual/repeated 404; controller-UID observer confirms owned job/cluster/batch Job/pod/Workload disappearance and observed GPU pods gone; PVC UIDs/output preserved, shared RayJob API survives |
| DEL-04/07/08 (deferred) | Active/shared/unowned/retrying/UID-race/finalizer cases remain unit-tested only; baseline does not inject these faults |
| OPS-01 (extended) | `deploy.py`: read-only preflight, Podman/Quay/Helm, fresh namespace/fixtures, unique release/tag, pushed digest, readiness/Route TLS, running image IDs/service versions, terminal summary/result evidence |
| OPS-02–09 (partial/deferred) | Bounded config/auth failures and independent invocations; restart, ambiguous-response injection, concurrency/load, upgrade and outages need focused follow-ups |
| TP | Disconnected, ROCm/CPU, operator/Trainer, estimation/custom code/new storage not introduced or certified |

Optional fixture skips are visible; a selected profile missing prerequisites fails
preflight. An all-skipped run fails. Smoke/unqueued-opt-out is **not** both
scheduling profiles or customer readiness. Attach actual sanitized reports and
component versions per environment; this table does not imply live execution.
