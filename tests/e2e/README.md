# Live Training Service E2E

Tracking: [RHOAIENG-102547](https://redhat.atlassian.net/browse/RHOAIENG-102547).
These tests call a real HTTP deployment, not TestClient or a mocked backend.
Manually invoked automation, not Jenkins/CI; `make test` and `make check` remain
unit-only. See the [full plan](test-plan.md) and [coverage/gaps](coverage.md).

## Build, push, deploy and test — one command

Prepare once:

- Install `uv`, Podman, Helm 3.13+ and `oc`; start your Podman machine if needed.
- `oc login` to your existing Ray-enabled test cluster; `podman login quay.io`.
  Never enable shell tracing or copy tokens into files.
- Use an existing **dedicated namespace**. The deployment identity needs Helm
  install/uninstall permissions, namespace/node reads, Route/service/pod reads,
  pod exec for version evidence, and permission to install the chart's
  TokenReview/SAR ClusterRole and ClusterRoleBinding. This is separate from the
  HTTP caller token below; admin credentials cannot prove restricted-user isolation.
- Use your own Quay repo. If private, prepare a namespace pull Secret and reference
  it through `imagePullSecrets`. The runner never reads/copies registry credentials.
- Copy [values.example.yaml](values.example.yaml) outside the repo and supply a
  compatible Ray + Training Hub runtime, authenticated Ray gateway and public CA.
  The chart's optional `volumes`/`volumeMounts` mount an existing CA ConfigMap.
  This PR does not change SDK/Ray pins or establish a new compatibility matrix.

From the repository root:

```bash
export TRAINING_SERVICE_TOKEN="$(oc whoami -t)"
# Only for a Route signed by a private CA (public certificate, not a token):
export TRAINING_SERVICE_CA_BUNDLE=/path/to/ingress-ca.pem
make test-e2e-deploy E2E_ARGS="--image-repository quay.io/YOUR_ORG/training-service --namespace YOUR_TEST_PROJECT --values /path/to/e2e-values.yaml"
```

The runner validates prerequisites, builds the existing `Containerfile` with Podman,
pushes a unique `e2e-<commit>-<run>` tag, pins Helm to the **pushed digest**, waits
for readiness/TLS, discovers the Route and executes HTTP sanity tests. No manual
build/push/deploy/test sequence is needed. Default `--platform linux/amd64` works
for AMD64 clusters, including from ARM Macs; use `linux/arm64` for ARM nodes.
Podman must support that platform (emulation can be slow).

Each invocation installs a new `training-e2e-<run>` release. It never adopts an
existing release, creates a namespace, changes DSC/queues, installs operators or
provisions storage. Smoke includes negative POSTs using nonexistent references:
it is **not strictly read-only**, but allocates no compute on a correct service.
A configured training runtime is required.

Successful deployments are uninstalled with a bounded wait; `--keep-deployment`
retains one for manual inspection. Failed/interrupted deployments are retained
with their exact cleanup command. Quay tags and training outputs are retained;
remove test images later under your registry retention policy. No registry image
or customer storage deletion is automated.

## Real training: Kueue absent and enabled

Add a small known-good [workload JSON](job.example.json). Replace all sample
PVC/model/dataset/output references. Project-local storage and HF/S3 Secrets must
already exist. Main currently supports basic `lora_sft`, not all Training Hub APIs.

```bash
make test-e2e-deploy E2E_ARGS="--image-repository quay.io/YOUR_ORG/training-service --namespace YOUR_TEST_PROJECT --values /path/to/e2e-values.yaml --job-file /path/to/tiny-job.json --queue-profile no-kueue --artifact-pod e2e-output-verifier"
```

- `no-kueue`: **omit `queue_name` entirely**. Preflight requires Kueue's API group
  to be absent. This conservative check does not certify installed-but-disabled
  Kueue; use a separately prepared absent-Kueue environment for that evidence.
- `with-kueue`: supply a nonempty `queue_name` in the JSON. Prepare Ray/Kueue DSC
  settings, RayJob integration, namespace enrollment, ResourceFlavor, active
  ClusterQueue/quota and LocalQueue. Verify the label and an owner-linked
  Workload with `Admitted=True`, not just successful POST.
- `unqueued-opt-out`: omit `queue_name` only where project policy explicitly
  permits it. It is recorded separately and **does not prove Kueue absence**.
  A default LocalQueue silently selected by the SDK fails unqueued proof.
  The runner never disables mandatory admission.

Repeat with `--queue-profile with-kueue` and a queued fixture on the prepared
Kueue environment. Both profiles need independent passing evidence before handoff.

Training must reach `SUCCEEDED`, preserve the submitted request/submitter, appear
in paginated/status-filtered reads, return driver snapshots and deliver two
distinct SSE events. `202`/`FAILED` never counts as success. Use a workload long
enough to observe streaming and running owned GPU pods; fast-disappearing clusters
fail observation, rather than manufacture evidence. SSE checks incremental framing/
delivery, not that every line was generated after opening the connection.

The runner enables log checks and the cluster observer for training. The observer
uses **kubeconfig**, never the HTTP caller token; it cannot make a denied API call
succeed. It needs namespace get, PVC get, RayJob get/list, RayCluster list, batch
Job list and pod get/list. With Kueue also: Workload list, LocalQueue get and
ClusterQueue get. Inventory is paginated. Transitive controller-owner UIDs link
the RayJob/clusters/batch Jobs/pods. After terminal DELETE the observer waits for
owned resources/Workload to disappear and verifies observed running GPU pods are
gone. Global free-GPU counts need not rise if other workloads consume capacity.

### Verify saved output independently

`--artifact-pod` selects a prepared Python-enabled pod (default container
`verifier`) mounting the **complete output PVC read-only**, without subPath. See
[artifact-verifier.example.yaml](artifact-verifier.example.yaml). Prepare it once
with your approved image/PVC; the runner does not create it. The observer requires
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

Names are short UUIDs; output prefixes gain that name. Fixtures are never modified.
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
