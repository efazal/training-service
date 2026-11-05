# Live Training Service E2E

Tracking: [RHOAIENG-102547](https://redhat.atlassian.net/browse/RHOAIENG-102547).
These tests call a real HTTP deployment, not TestClient or a mocked backend.
Manually invoked automation, not Jenkins/CI; `make test` and `make check` remain
unit-only. See [coverage/gaps](coverage.md) and [examples](examples/).

## Build, push, deploy and test — one command

Prepare once:

- Install `uv`, Podman, Helm 3.13+ (including Helm 4) and `oc`; start your Podman machine if needed.
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
- The deployment identity needs namespace creation/deletion, Helm install/uninstall,
  StorageClass/node reads, Route/service/pod reads, `get` access to ConfigMap
  `openshift-config-managed/default-ingress-cert` for public TLS trust discovery,
  pod exec for evidence, and permission to install the chart's
  TokenReview/SAR ClusterRole and ClusterRoleBinding (list/delete for cleanup verification).
  This is separate from the
  HTTP caller token below; admin credentials cannot prove restricted-user isolation.
- Use your own Quay repo. Public images need no credential transfer. For a private
  repo, add `--use-podman-auth` as described below: local Podman login alone does
  not authorize Kubernetes image pulls. The runner never copies a pre-existing
  cluster Secret or changes cluster-wide pull credentials.
- Copy [values.example.yaml](examples/values.example.yaml) outside the repo and supply
  the authenticated Ray gateway. The chart's public Ray + Training Hub image is
  used by default; set `trainingRuntime.image` only to override it.
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
make test-e2e-deploy E2E_ARGS="--image-repository quay.io/YOUR_ORG/training-service --values /path/to/e2e-values.yaml --storage-class YOUR_RWX_CLASS"
```

This keeps PR #10's baseline `lora_sft` lifecycle. To run the **four-algorithm
matrix** on the same isolated deployment, add `--algorithm-matrix` to `E2E_ARGS`.
It submits separate `sft`, `osft`, `lora_sft` and `lora_grpo` jobs, checks
accepted requests and preserved parameters, waits for `SUCCEEDED`, reads log snapshots,
checks output files and verifies owner-scoped cleanup. The fixture prepares chat
and math JSONL datasets. GRPO needs a GPU-free Ray head plus a GPU worker, so
the ClusterQueue must admit the larger request. The selected Ray/Training Hub
runtime must actually support all four backends; the test does not install or
repair runtime dependencies. A known verl/TRL import failure is reported as a
failed GRPO test, never counted as a pass.
The baseline run still verifies incremental SSE. The matrix omits that timing-
sensitive check so a short training run cannot fail solely because it finished
before a stream could be observed.

```bash
make test-e2e-deploy E2E_ARGS="--image-repository quay.io/YOUR_ORG/training-service --values /path/to/e2e-values.yaml --storage-class YOUR_RWX_CLASS --algorithm-matrix"
```

Against an existing deployment, use `make test-e2e` with `--algorithm-matrix`
and `--job-file` pointing to a PVC template whose dataset is under `datasets/`.
Stage `tiny-chat.jsonl` and `tiny-math.jsonl` in that same PVC first. The
matrix's algorithm-specific parameters are in
[`algorithms.py`](algorithms.py); each request is stored unchanged in the job
detail. This is representative end-to-end coverage, not an exhaustive test of
every optional Training Hub parameter.

### Remote model/dataset inputs and S3 output

`test_remote_storage.py` is an **opt-in live E2E matrix** for all four input
combinations: S3/S3, S3/HF, HF/S3 and HF/HF (model/dataset). Each test submits
`lora_sft` through the HTTP API, waits for `SUCCEEDED`, deletes its terminal
RayJob, then reads back a valid LoRA adapter config and safetensors header from
its unique S3 output prefix. A separate HTTP-negative test rejects HF as an
output destination: the API contract supports S3 or PVC output, not HF publishing.
This is not a mock and a `202` alone does not pass the test.

Copy [`remote-jobs.example.json`](examples/remote-jobs.example.json) **outside
the repository** and edit it for the target project. The named S3 connections
must be project-local Secrets containing `AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`
and `AWS_SECRET_ACCESS_KEY` (optional `AWS_DEFAULT_REGION`/`AWS_SESSION_TOKEN`).
The HF Secret needs `HF_TOKEN` or `token`, even for public repositories. Stage a
small model snapshot in the S3 model prefix and a small chat JSONL file (one
`messages` list of `user`/`assistant` turns per line) at the S3 dataset key. Set
the HF dataset to a small `train` split with the same `messages` column. Alpaca
data fails: `lora_sft` trains with `assistant_only_loss`, which needs chat
turns. The two models/datasets need to be
compatible with the fixture's LoRA parameters and runtime. The example names
and repositories are illustrative, **not pre-provisioned fixtures**. The API
caller must be authorized to read the named Secrets and create RayJobs.

Use a verifier identity with S3 read access to the **same output bucket**. It is
separate from the project-local connection Secret, and credentials are supplied
only through environment variables, never JSON or test reports:

```bash
export TRAINING_SERVICE_URL='https://YOUR_TRAINING_SERVICE_ROUTE'
export TRAINING_SERVICE_PROJECT='YOUR_PROJECT'
# Use an existing oc login, or set TRAINING_SERVICE_TOKEN explicitly.
export TRAINING_S3_E2E_ENDPOINT='https://YOUR_S3_ENDPOINT'
export TRAINING_S3_E2E_BUCKET='my-test-bucket'
export TRAINING_S3_E2E_ACCESS_KEY='...'
export TRAINING_S3_E2E_SECRET_KEY='...'
uv run --locked --with boto3==1.40.61 pytest tests/e2e/test_remote_storage.py \
  --remote-fixtures-file /path/outside/repo/remote-jobs.json
```

Optional verifier settings: `TRAINING_S3_E2E_REGION`,
`TRAINING_S3_E2E_SESSION_TOKEN`, `TRAINING_S3_E2E_CA_BUNDLE`. The test never
creates Secrets or copies credentials into the cluster. It deletes only its own
terminal API jobs; S3 output objects remain in the unique per-job prefix for
inspection and should be removed separately when no longer needed. Without the
fixture flag, these four tests skip visibly. Missing verifier settings fail
preflight rather than silently skipping selected remote tests.

Both E2E commands use `oc whoami -t` from your current login when
`TRAINING_SERVICE_TOKEN` is unset or blank. No manual token export is needed.
Set `TRAINING_SERVICE_TOKEN` explicitly to use another API caller, or to test an
existing service on another cluster without changing your `oc` login. An explicit
token always wins; an invalid/expired override is never silently replaced.
Run `unset TRAINING_SERVICE_TOKEN` to remove an old override and use your current
login again. Tokens are captured in memory, never printed or saved to reports.
The token must be valid for the cluster hosting the service; overriding it does
not change the kubeconfig context used for deployment or cluster observation.

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

To rerun using an already-built service image, add
`--reuse-image quay.io/YOUR_ORG/training-service@sha256:YOUR_DIGEST` in the same
repository as `--image-repository`. This skips Podman build/push and does not
require a running Podman machine or Quay push login. The cluster must still be
able to pull that image. The digest is recorded as reused, not a new build.
Only reuse it when the service source, dependencies, `Containerfile` and other
files copied by the `Containerfile` are unchanged; an unchanged `Containerfile`
alone is insufficient. Test-harness-only changes do not require an image rebuild.
Select a known-compatible image explicitly; there is no silent latest-tag cache.

### Private Quay images

Add `--use-podman-auth` to the deploy command (also supported with `--reuse-image`):

```bash
make test-e2e-deploy E2E_ARGS="--image-repository quay.io/YOUR_ORG/training-service --values /path/to/e2e-values.yaml --storage-class YOUR_RWX_CLASS --use-podman-auth"
```

This is explicit permission to send your **Quay credential** to the test cluster.
Prefer a read-only robot account with access only to the test repositories when
reusing an image. Cluster administrators and identities permitted to read Secrets
in the test namespace can access that credential. The runner honors the existing
Kubernetes API TLS settings in kubeconfig, just like `oc`; it does not modify
them. If `insecure-skip-tls-verify` is already enabled, it warns and continues:
the API-server identity is not verified when sending registry credentials. Use
that setting only for a trusted development cluster; verified TLS is recommended.
Kubernetes HTTP debug logging remains blocked to prevent credential disclosure.
Route/gateway certificate verification is unchanged. Secret get/create/delete and
default ServiceAccount get/patch permissions are required in the new namespace.

The runner reads Podman's normal local auth locations (or the file selected by
`REGISTRY_AUTH_FILE`) and resolves configured native credential helpers. It copies
**only the `quay.io` login**, not other registry entries, into an in-memory
`kubernetes.io/dockerconfigjson` Secret request. Unsupported/missing credentials
fail with login guidance, not raw helper/API output. Leave `imagePullSecrets: []`
in values; the runner supplies the newly created Secret name to Helm, the
fixture/verifier pod and the new namespace's default ServiceAccount. Default-SA
Ray/submitter pods inherit the reference; custom runtime ServiceAccounts need
their own approved pull configuration. This flag supports Quay, not arbitrary
registries or per-repository auth-file entries. Images using a different registry
must already be pullable in the cluster.

No credential file or Secret manifest is written to the repository, evidence,
Helm values/history or command arguments. Only the Secret **name** is supplied to
Helm. The runner checks namespace/Secret ownership and uses a UID precondition to
remove its Secret on success, failure or interruption, even with
`--keep-resources`. Its default-ServiceAccount reference is removed too; other
Secrets/references are untouched. A retained private deployment may need fresh
credentials before it can pull again. Cleanup failure is reported and makes the
run fail; inspect the named test-only Secret. Uncatchable termination or machine
loss can leave it behind, so verify cleanup in that case. No credentials are read
when this flag is omitted.

Each invocation creates a new `training-e2e-<run>` namespace and Helm release.
The [namespace manifest](fixtures/namespace.yaml) includes
`kueue.openshift.io/managed: "true"`, `opendatahub.io/dashboard: "true"`, and a
unique E2E ownership label. It creates LocalQueue `e2e` backed by the existing
ClusterQueue `default` (override with `--cluster-queue`), a 5Gi RWX PVC and a
fixture/verifier pod. The pod downloads a pinned public Qwen2.5-0.5B-Instruct
revision and prepares eight chat examples plus twelve math examples; training uses PVC references and
needs no HF token. Set `trainingRuntime.image` in the E2E values file to the
tested Ray/Training Hub image; it is separate from the API service image the runner
builds or reuses. The runner exports that runtime's Training Hub registry into
a test-owned ConfigMap, configures algorithm discovery in Helm, and checks the
compact list, details, schema opt-in, native parameter examples and error responses. The generated
request and manifests (including the catalog) are saved with results.
The fixture pod uses the new namespace's OpenShift-allocated `fsGroup` so the CSI
driver can grant non-root access to its new PVC; no privileged/root setup pod or
shared storage permission change is used. A write check runs before the download.
Readiness polling prints progress and immediately reports failed init containers,
including a targeted `oc logs` command, instead of waiting the full fixture timeout.
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

On success, failure or interruption, the runner **cleans up by default**: uninstall
the isolated Helm release (including its cluster-scoped RBAC), then delete the
run-owned namespace, LocalQueue, verifier, temporary credentials, test PVC and
remaining Ray resources. Namespace ownership/run label and UID are verified;
deletion uses a UID precondition and bounded polling, never forced finalizer removal.
Partial setup failures also trigger cleanup. This may cancel a still-running test
job in this isolated namespace; it does not change the API's terminal-only DELETE
contract. Cleanup failure makes the run fail and reports manual recovery guidance.
If Helm uninstall fails, the namespace/release history is retained for safe recovery.
PVC deletion follows the StorageClass reclaim policy; the runner never directly
deletes PVs or changes shared storage policy.

Add **`--keep-resources`** to retain the namespace, deployment and test storage for
debugging; `--keep-deployment` remains a compatibility alias. Temporary registry
credentials are still removed. Retained active jobs can consume GPUs: stop them
and uninstall the reported Helm release before deleting that test-only namespace.
Reports, manifests and JUnit/JSON evidence always remain in the local results directory.
Test images remain in Quay under your registry retention policy. No registry image,
pre-existing namespace, customer storage or shared CRD deletion is automated.

On a pytest deadline/interruption, send pytest SIGINT (its handled interrupt) and
allow `--cleanup-timeout` for fixture teardown before killing a stuck child. The
parent then performs the same owned-namespace cleanup. SIGTERM uses this handled
interruption path too. A second interrupt stops/reaps the pytest child; interrupting
resource cleanup reports a cleanup failure rather than a passing run. Uncatchable termination
(SIGKILL or machine loss) cannot run finalizers; inspect the reported namespace.

## Additional profiles against an existing deployment

The normal deployment command above **always includes Kueue**. To separately
test an already-deployed service without Kueue, use `make test-e2e`, not the
deployment runner. Supply a known-good [workload JSON](examples/job.example.json), existing
PVC/model/dataset/output references and a prepared
[read-only verifier pod](examples/artifact-verifier.example.yaml). Use discovery to
check the algorithms and parameters available in the configured runtime. The default
sanity workload uses `lora_sft`; `--algorithm-matrix` opts into the four supported
execution profiles. Python-only Training Hub modes are not supported.

```bash
# Set TRAINING_SERVICE_URL and TRAINING_SERVICE_PROJECT; the token defaults to oc.
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
[artifact-verifier.example.yaml](examples/artifact-verifier.example.yaml). Prepare it once
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
| `--reuse-image IMAGE@sha256:DIGEST` | Reuse a known-compatible image in `--image-repository`, skipping build/push |
| `--use-podman-auth` | Temporary namespace-local Quay pull credentials from Podman; honors kubeconfig TLS and warns if verification is disabled |
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
reported. Unset any explicit `TRAINING_SERVICE_TOKEN` override when finished.
