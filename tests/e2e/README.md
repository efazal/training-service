# Live Training Service E2E

Tracking: [RHOAIENG-102547](https://redhat.atlassian.net/browse/RHOAIENG-102547).
These tests call a real HTTP deployment, not TestClient or a mocked backend.
Manually invoked automation, not Jenkins/CI; `make test` and `make check` remain
unit-only. See the [full plan](test-plan.md) and [coverage/gaps](coverage.md).

## Build, push, deploy and test — one command

Prepare once:

- Install `uv`, Podman, Helm 3.13+ and `oc`; start your Podman machine if needed.
- `oc login` to your test cluster; `podman login quay.io`.
  Never enable shell tracing or copy tokens into files.
- An administrator prepares Ray and Kueue, including Kueue's RayJob integration,
  ResourceFlavors and an Active ClusterQueue with CPU, memory and NVIDIA GPU
  quota. **Direct installations are supported**: the runner does not read the DSC
  or require Kueue to be DSC-managed. It assumes this infrastructure is ready;
  the queued test verifies actual APIs, namespace policy and admission. A Kueue
  problem fails that test with diagnostics, not the independent API tests.
- Select a **ReadWriteMany-capable StorageClass**, writable by OpenShift-assigned
  nonroot UIDs. The built-in fixture needs one NVIDIA GPU, 4 CPUs, 32Gi memory,
  5Gi storage and outbound access to the public Hugging Face model repository.
  This is not a disconnected/ROCm or CPU-only test.
- The deployment identity needs namespace creation, Helm install/uninstall,
  StorageClass/node reads, Route/service/pod reads, `get` access to ConfigMap
  `openshift-config-managed/default-ingress-cert` for public TLS trust discovery,
  pod exec for evidence, and permission to install the chart's
  TokenReview/SAR ClusterRole and ClusterRoleBinding. This is separate from the
  HTTP caller token below; admin credentials cannot prove restricted-user isolation.
- Use your own **public** Quay repo, or cluster-wide image pull credentials.
  Every run creates a fresh namespace; the runner never reads/copies a pull Secret
  from another namespace. Both service and runtime images must be pullable there.
- Copy [values.example.yaml](values.example.yaml) outside the repo and supply a
  compatible Ray + Training Hub runtime and authenticated Ray gateway.
  The runner automatically reads the cluster's **public ingress certificate
  bundle** from `openshift-config-managed/default-ingress-cert` (`ca-bundle.crt`).
  It saves it only under the ignored per-run results directory, uses it for
  Route TLS, and provisions `training-service-ray-ca` for the gateway using the
  example's read-only mount. No PEM needs to be prepared or committed; no Secret
  or private key is read. Discovery uses the authenticated `oc` connection,
  never an unverified peer certificate.
  This PR does not change SDK/Ray pins or establish a new compatibility matrix.

From the repository root:

```bash
export TRAINING_SERVICE_TOKEN="$(oc whoami -t)"
make test-e2e-deploy E2E_ARGS="--image-repository quay.io/YOUR_ORG/training-service --values /path/to/e2e-values.yaml --storage-class YOUR_RWX_CLASS"
```

For a Ray gateway with a certificate from a different issuer, override only its
trust with `--ray-ca-bundle /path/to/custom-ray-ca.pem`. To override Route trust,
set `TRAINING_SERVICE_CA_BUNDLE=/path/to/custom-ingress-ca.pem`; this does not
override Ray trust. Unset stale overrides to restore automatic discovery.
For a publicly trusted Ray gateway, remove the example's CA environment/volume
entries to use the container's system trust store. If ConfigMap discovery is
denied/missing, the runner fails with permission/override guidance; it never
disables certificate or hostname verification. Both explicit overrides together
(or just the Route override when Ray uses system trust) avoid the ConfigMap read.

The runner validates prerequisites, builds the existing `Containerfile` with Podman,
pushes a unique `e2e-<commit>-<run>` tag, pins Helm to the **pushed digest**, waits
for readiness/TLS, discovers the Route and executes HTTP sanity **and real
Kueue-admitted training**. No manual namespace, queue, PVC, dataset, model download,
verifier pod, build/push/deploy/test sequence is needed. Default
`--platform linux/amd64` targets AMD64 clusters, including from ARM Macs; use
`linux/arm64` for ARM nodes.
Podman must support that platform (emulation can be slow).

Each invocation creates a new `training-e2e-<run>` namespace and Helm release.
The [namespace manifest](fixtures/namespace.yaml) includes
`kueue.openshift.io/managed: "true"`, `opendatahub.io/dashboard: "true"`, and a
unique E2E ownership label. It creates LocalQueue `e2e` backed by the existing
ClusterQueue `default` (override with `--cluster-queue`), a 5Gi RWX PVC and a
fixture/verifier pod. The pod downloads a pinned public Qwen2.5-0.5B-Instruct
revision and prepares eight chat examples; training uses PVC references and
needs no HF token. The generated request and manifests are saved with results.
The ClusterQueue selector must admit these labels. The runner never changes the
DSC, operators, ClusterQueue quota or ResourceFlavors.

Add `--namespace NEW_NAME` to choose the new namespace. An existing namespace is
always rejected, never adopted. The normal deployment run has no smoke-only or
no-Kueue switch: admission, training, logs, outputs and cleanup are required.

To check prerequisites without building, uploading or creating cluster resources:

```bash
make test-e2e-deploy E2E_ARGS="--image-repository quay.io/YOUR_ORG/training-service --values /path/to/e2e-values.yaml --storage-class YOUR_RWX_CLASS --preflight-only"
```

The terminal summary reports pass/fail/skip counts, training success, Kueue
admission, GPU cleanup, checkpoint verification and evidence paths. Kueue setup
and test diagnostics also appear in the summary and JSON/JUnit evidence. Missing
APIs/queues, denied observer permissions, inactive queues, namespace-selector
mismatches and missing resource coverage have distinct diagnostics; training
timeouts include the last observed Workload condition reasons. Raw controller
messages and API error bodies are not copied into reports.

A LocalQueue creation/check failure does not abort deployment or the suite.
The queued test fails before submission if its prerequisites are unavailable;
independent tests continue. The deployment runner overrides inherited pytest
fail-fast settings (`--maxfail=0`) and **still exits nonzero** for failed tests.
It never silently skips queued training or falls back to unqueued execution.
Missing required lifecycle evidence fails even if pytest was accidentally filtered.
`--preflight-only` checks deployment/tooling/TLS prerequisites, **not Kueue
readiness**; actual queue checks belong to the selected lifecycle test.

Successful deployments and the fixture/verifier pod are removed with bounded
waits; `--keep-deployment` retains both for manual inspection. Failed/interrupted
deployments and fixtures are retained for diagnosis. The **namespace, LocalQueue,
PVC, outputs and Quay tag are retained**, and each run creates new ones;
review these test-only resources periodically rather than expecting automatic
namespace/storage deletion. The exact Helm cleanup command is printed on failure.
Remove test images later under your registry retention policy. No registry image
or customer storage deletion is automated.

## Additional profiles against an existing deployment

The normal deployment command above **always includes Kueue**. To separately
test an already-deployed service without Kueue, use `make test-e2e`, not the
deployment runner. Supply a known-good [workload JSON](job.example.json), existing
PVC/model/dataset/output references and a prepared
[read-only verifier pod](artifact-verifier.example.yaml). Main currently supports
basic `lora_sft`, not all Training Hub APIs.

```bash
# Set TRAINING_SERVICE_URL, TRAINING_SERVICE_PROJECT and TRAINING_SERVICE_TOKEN first.
make test-e2e E2E_ARGS="--job-file /path/to/tiny-job.json --queue-profile no-kueue --observe-cluster --artifact-pod e2e-output-verifier --check-logs --check-stream -s"
```

- `no-kueue`: **omit `queue_name` entirely**. Preflight requires Kueue's API group
  to be absent. This conservative check does not certify installed-but-disabled
  Kueue; use a separately prepared absent-Kueue environment for that evidence.
- `with-kueue`: supply a nonempty `queue_name` in the JSON. Prepare Ray/Kueue
  (DSC-managed or directly installed), RayJob integration, namespace enrollment, ResourceFlavor, active
  ClusterQueue/quota and LocalQueue. Verify the label and an owner-linked
  Workload with `Admitted=True`, not just successful POST.
- `unqueued-opt-out`: omit `queue_name` only where project policy explicitly
  permits it. It is recorded separately and **does not prove Kueue absence**.
  A default LocalQueue silently selected by the SDK fails unqueued proof.
  The runner never disables mandatory admission.

Both scheduling configurations need independent passing evidence before
advertising support for both; the default Kueue run does not certify Kueue absence.

Training must reach `SUCCEEDED`, preserve the submitted request/submitter, appear
in paginated/status-filtered reads, return driver snapshots and deliver two
distinct SSE events. `202`/`FAILED` never counts as success. Use a workload long
enough to observe streaming and running owned GPU pods; fast-disappearing clusters
fail observation, rather than manufacture evidence. SSE checks incremental framing/
delivery, not that every line was generated after opening the connection.

The deployment runner always enables log checks and the cluster observer. The observer
uses **kubeconfig**, never the HTTP caller token; it cannot make a denied API call
succeed. It needs namespace get, PVC get, RayJob get/list, RayCluster list, batch
Job list and pod get/list. With Kueue also: Workload list, LocalQueue get and
ClusterQueue get. Inventory is paginated. Transitive controller-owner UIDs link
the RayJob/clusters/batch Jobs/pods. After terminal DELETE the observer waits for
owned resources/Workload to disappear and verifies observed running GPU pods are
gone. Global free-GPU counts need not rise if other workloads consume capacity.

### Verify saved output independently

The deployment runner creates its verifier automatically. For an existing
deployment, `--artifact-pod` selects a prepared Python-enabled pod (default container
`verifier`) mounting the **complete output PVC read-only**, without subPath. See
[artifact-verifier.example.yaml](artifact-verifier.example.yaml). Prepare it once
with your approved image/PVC. The observer requires
pod get and pods/exec create. Exec only reads the unique output directory.

Current LoRA verification checks nonempty LORA config and safetensors headers/
tensors/data bounds, before **and after deletion**, and preserves PVC UIDs. This
is checkpoint-format validation, not model-quality or inference evaluation; it
does not certify other artifact formats. Without `--artifact-pod`, results say
outputs were **not independently verified**, not a complete handoff gate.
Pre-existing Secrets, shared CRDs and outputs are never deleted.

## Test an existing deployment

```bash
export TRAINING_SERVICE_URL=https://training-service.example.com
export TRAINING_SERVICE_PROJECT=YOUR_TEST_PROJECT
export TRAINING_SERVICE_TOKEN="$(oc whoami -t)"
make test-e2e
```

Use an origin **without `/api/v1`**. TLS verification is mandatory; HTTP only on
loopback port-forwarding. Redirects and environment HTTP proxies are disabled.
POST is **never automatically retried**. For complete training checks:

```bash
make test-e2e E2E_ARGS="--job-file /path/to/tiny-job.json --queue-profile with-kueue --observe-cluster --artifact-pod e2e-output-verifier --check-logs --check-stream -s"
```

| Option | Purpose |
| --- | --- |
| `--job-timeout 1800` | Training deadline, seconds |
| `--stream-timeout 60` | SSE deadline; socket reads bounded too |
| `--cleanup-timeout 180` | Terminal/resource cleanup deadline |
| `--request-timeout 30` | Per HTTP timeout, existing-deployment command |
| `--build-timeout 1800`, `--deploy-timeout 300` | Build/push and rollout/Route deadlines, deployment runner |
| `--fixture-timeout 900` | PVC/model preparation deadline, deployment runner |
| `--namespace NEW_NAME`, `--cluster-queue existing-cq` | New namespace name and existing ClusterQueue override |
| `--preflight-only` | Read-only cluster/tooling/config checks; no build/push/deploy |
| `--ray-ca-bundle /path/to/custom-ray-ca.pem` | Override automatic ingress trust for a custom Ray gateway |
| `--forbidden-project other-project` | Existing project the HTTP caller cannot access; not an admin-token test |
| `--keep-failed-job` | Existing-deployment only: retain FAILED/STOPPED for inspection, still fail |
| `--junitxml /path/results.xml` | Optional sanitized report, existing-deployment command |

Prepared optional regressions, existing-deployment command only:

- `-k missing_ray_submission --missing-submission-job <terminal-job>`: both log
  modes must return JSON `404`, not retryable `409`; fixture cluster still exists.
- `-k invalid_kubernetes_url --expect-invalid-kubernetes-url`: use an **isolated**
  service configured with an invalid Kubernetes URL; expect sanitized `503`.
  Never break a shared deployment. The suite does not inject faults itself.

## Evidence, failure and recovery

Deployment runs write ignored `artifacts/e2e/<run>/`: JUnit, outcome/skip counts,
commit/dirty status, digest and running image IDs, service versions, chart/OpenAPI
checksums and lifecycle/queue/GPU/artifact evidence. Runtime Training Hub/operator
versions not independently observed must be recorded separately for the release
matrix, not inferred from tags. Smoke/skipped checks do not certify readiness;
all-skipped runs fail. [coverage.md](coverage.md) lists remaining gaps.
Discovered public certificates and their ConfigMap manifest stay in this ignored
results directory; `run.json` records trust sources and the ingress bundle checksum.

Names are short UUIDs; output prefixes gain that name. Existing-deployment input
fixtures are never modified; the deployment runner writes only its new namespace/PVC.
Cleanup registered before POST reconciles lost responses by name without replay.
It refuses mismatched stored submissions, waits for terminal status and uses only
API DELETE. Timeouts/token expiry/interruption report resources needing inspection.
Never cancel running jobs, force-remove pods/finalizers, delete PVCs/Secrets/CRDs
or change queue quota to manufacture a pass.

Inspect the printed job ID using GET/logs, wait for terminal, DELETE through the
API, then verify owned pods/clusters disappear before uninstalling the test service.
Repeat twice with identical fixtures to check independent job/output identities.
No `set -x`, pytest `--showlocals`, HTTP debug logging, sensitive fixtures or token
CLI arguments. Raw build/Helm/cluster output is withheld; stages/exit status are
reported. Unset `TRAINING_SERVICE_TOKEN` when finished.
