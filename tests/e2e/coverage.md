# Coverage and customer-readiness gaps

Source: [RHOAIENG-102547](https://redhat.atlassian.net/browse/RHOAIENG-102547).

This PR is a repeatable baseline, **not a claim that every P0 or configuration
passes**. Mocked unit tests are not live evidence.

The deployment workflow provides Podman → Quay → Helm automation, a fresh labeled
namespace, LocalQueue/PVC/fixture provisioning, and mandatory Kueue-admitted
training. Shared cluster/operator/DSC/ClusterQueue/StorageClass provisioning and
Jenkins remain excluded; users prepare Ray/Kueue first. Direct installation is
supported without DSC checks. Kueue failures fail the queued test with diagnostics
while independent API tests continue; the final run still fails.
For `test-e2e-deploy`'s newly created, UID/run-label-owned namespace, default
cleanup runs after success/failure/interruption, including test PVCs and leftover
Ray resources. `--keep-resources` retains debugging resources, never temporary
registry credentials. Existing-deployment tests preserve customer namespaces/data.

| Plan IDs | Implemented test / remaining status |
| --- | --- |
| RHOAIENG-96354 | Real-runtime ConfigMap catalog, compact list, details/schema opt-in, native parameter examples without generated POST requests, unknown algorithm, invalid query, missing/invalid bearer tokens. Empty/malformed catalog and K8s 401/403/404/outages are unit-tested; not injected live. |
| ERR-01–05/10 (partial) | `test_service.py`: envelope/media type, probes/OpenAPI, auth, malformed JSON/schema/media type, invalid queries/URI/resources/algorithm, missing job/project/PVC/Secret; duplicate during lifecycle |
| ERR-06–09 (partial/opt-in) | Prepared invalid Kubernetes URL regression; broader outages/faults and field-level guidance deferred |
| ALG-03/06/08 (partial) | Rejection matrix: unknown algorithm, bool-as-number, unsupported/invalid parameters, traversal; API confirms no job |
| ALG-01/02/04/05 (opt-in) | `--algorithm-matrix` runs real `sft`, `osft`, `lora_sft` and `lora_grpo` through the API with representative advanced overrides, status/log/output checks and cleanup. Unsupported/malformed overrides are also rejected per algorithm in the HTTP suite. Success requires a compatible runtime and actual passing reports; GRPO's known verl/TRL import failure is not hidden. |
| ALG-07/09/10 (conditional/deferred) | Discovery is still a separate feature. The opt-in matrix covers representative parameters, not every Training Hub option, runtime release or dataset format; attach live passing evidence for each backend. |
| Reported LoRA regression, ERR-07 | Original extra parameters are preserved. Per-algorithm HTTP rejection checks now require the algorithm name in validation errors; the matrix verifies representative parameters survive in the submitted request. |
| JOB-01/02/06, DATA-01/06 (partial) | Lifecycle: 202/Location, duplicate, request/submitter/resources, real SUCCEEDED; default deploy run requires read-only LoRA artifact-format check before/after delete, not inference evaluation/all storage providers |
| JOB-03/04/12 (partial) | Missing project/resources, rejected-job lookup, KFT 501 vs invalid backend 422. Independent cluster/storage no-side-effect proof for every negative case deferred |
| JOB-05/07–11 (partial/deferred) | Same-project duplicate covered; cross-project names, controlled runtime failure, dataset matrix and multi-node/GPU follow-ups |
| SEC-01/06/07 (partial) | Missing/invalid/malformed token, Bearer challenge, token echo guard, verified TLS/no redirects/proxies. Automatic public ingress CA discovery and separate explicit Route/Ray overrides; no Secret reads, private-key copies or TLS bypass. Expired/revoked personas, Secret scans and adversarial TLS deferred |
| SEC-02–05/08–10 (partial/deferred) | Optional forbidden-project list; complete real least-privilege endpoint/persona matrix, permission changes and concurrent isolation deferred |
| DATA-02–05/07–09 (partial/opt-in) | Missing HF/S3 Secret/PVC context and PVC traversal rejection; PVC positive fixture. New `--remote-fixtures-file` matrix submits all four S3/HF model/dataset combinations, requires real `SUCCEEDED`, then reads back LoRA adapter artifacts from S3 after API deletion. HF output is rejected by HTTP 422. These remote tests require externally staged Secrets/model/datasets and S3 verifier credentials; no passing live evidence is implied. Private/gated failures and access-mode matrix remain deferred. |
| QUEUE-01/02/09/10 (profile-dependent) | Default deploy run assumes installed/configured Ray/Kueue (no DSC check), creates a labeled fresh namespace and LocalQueue, then requires an owned admitted Workload, SUCCEEDED, artifacts and GPU cleanup. Queue failures are test-local, not a suite-wide abort. Existing-deployment absent-Kueue/unqueued-opt-out profiles remain separate |
| QUEUE-03–08/11 (partial/deferred) | Invalid resources/empty queue covered. Quota hold/release, mandatory-admission denial, disabled integrations and compatibility matrix require prepared scenarios |
| READ-01–04/06/07 (partial) | Compact summaries/count bounds, pagination, created-job inclusion/status filters, payload/optional truthful timestamps, malformed token; exact inventory count/default-status/filter binding matrix deferred |
| READ-05/08 (deferred) | External/failed-before-start records, metadata after API restart require prepared scenarios |
| LOG-01–05 (partial) | Snapshot and two distinct SSE events during training; missing job both modes, optional missing Ray submission, invalid queries, notready/gone/backend statuses. Post-header faults, tail/Unicode/large-log matrix deferred |
| LOG-06/07 (deferred) | Closing successful follow exercises disconnect; reconnect, slow consumer, lifetime/leaks and malicious ownership/gateway fixtures still required |
| DEL-01–03/05/06 (partial) | Terminal delete/eventual/repeated 404; controller-UID observer confirms owned job/cluster/batch Job/pod/Workload disappearance and observed GPU pods gone; PVC UIDs/output preserved, shared RayJob API survives |
| DEL-04/07/08 (deferred) | Active/shared/unowned/retrying/UID-race/finalizer cases remain unit-tested only; baseline does not inject these faults |
| OPS-01 (extended) | `deploy.py`: read-only preflight, Podman/Quay/Helm or explicit digest-pinned image reuse, fresh namespace/fixtures, unique release/tag, pushed/reused digest, namespace-allocated fixture fsGroup and write check, progress/failed-init detection, readiness/Route TLS, running image IDs/service versions, terminal summary/result evidence; owned Helm/namespace/PVC cleanup by default, handled pytest interrupt/grace period, explicit resource-retention option |
| OPS-02–09 (partial/deferred) | Bounded config/auth failures and independent invocations; restart, ambiguous-response injection, concurrency/load, upgrade and outages need focused follow-ups |
| TP | Disconnected, ROCm/CPU, operator/Trainer, estimation/custom code/new storage not introduced or certified |

Optional fixture skips are visible; a selected profile missing queue prerequisites fails
its lifecycle test setup, not the entire suite. An all-skipped run fails. Smoke/unqueued-opt-out is **not** both
scheduling profiles or customer readiness. Attach actual sanitized reports and
component versions per environment; this table does not imply live execution.

Private Quay images: `--use-podman-auth` opts into a temporary namespace-local
pull Secret from Podman's Quay credential. Unit tests cover auth-file/helper
resolution, unrelated-credential exclusion, kubeconfig TLS preservation/warnings,
debug logging guards, Secret UID
ownership, ambiguous-create cleanup and cleanup on failed/retained workflows.
The deploy runner checks service-pod pull-secret references and reports credential
cleanup. These plumbing checks with public images alone do not certify actual
private pulls; use a private repository for that live evidence. No cluster-wide
credential or registry visibility changes are made.
