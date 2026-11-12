# PR #14 customer installation verification

Verified on 12 November 2026 against the existing OpenShift AI cluster in
`test-efazal`. The new release is `training-service`; existing releases were
preserved. These results cover the selected acceptance path, not every supported
algorithm, storage provider, authorization policy or failure scenario.

## Published artifacts

- API image: `quay.io/modh/training-service:0.1.0` (Linux AMD64).
- Image digest: `sha256:ae920c93d135dcc841c30a269f47358b1649ffe65b6aaf43c9dd64619949a2fb`.
- Application source: `994e168e919e01bfc3d215529a53493cba762cc9`.
- Chart: `training-service-0.1.0.tgz`.
- Chart SHA256: `550e137bf758b61e7e6d23714995bc9d5fe7fb537aebf429162e380e764709f4`.
- [Public Helm repository](https://github.com/opendatahub-io/training-service/tree/helm-charts), publication commit `c6411993cc7a948e269d40a1ffe4e8238f4ed5f3`.

Anonymous registry manifest access and anonymous chart download both returned
HTTP 200. The downloaded chart checksum and running pod image digest matched
the published artifacts. A native AMD64 pod imported the application, CodeFlare
SDK 0.40.1, Ray 2.58.0 and datasets 4.8.5 successfully; `pip check` also passed.
An earlier local AMD64-emulated application smoke hit a QEMU crash on the ARM
Mac; the native cluster checks above passed.

## Customer commands actually run

Using empty, isolated Helm configuration, cache and data directories:

```bash
helm repo add training-service https://raw.githubusercontent.com/opendatahub-io/training-service/helm-charts/
helm repo update training-service
helm install training-service training-service/training-service \
  --namespace test-efazal --create-namespace --wait --timeout 15m
```

No checkout, values file, `--set`, gateway URL, CA ConfigMap or registry login
was supplied to this installation. It targets an already prepared OpenShift AI
platform with installer RBAC permissions; the chart does not install Ray/Kueue,
GPU operators, queues or training data.

```text
STATUS: deployed
REVISION: 1
DESCRIPTION: Install complete
helm get values training-service -n test-efazal -o json: null
API pod: Ready, zero restarts
Catalog init container: exit 0, discovered 7 algorithms
```

The default image, runtime, HTTPS Route, authentication RBAC and automatic
gateway/ingress-CA discovery were used. Cluster-wide RBAC names include the
release namespace to coexist with other installations. A regression test failed
before this naming fix and passed afterward. The chart in the PR matches the
published package's file contents.

## Functional evidence

The live E2E suite passed **144 tests**, with **18 explicitly skipped**, in
**294.58 seconds**. It used the caller's current OpenShift login; credentials
were never saved in this evidence. The external API client verified Route TLS.
The independent cluster observer inherited the current kubeconfig's TLS settings,
which produced unverified-HTTPS warnings for its Kubernetes API reads.

| Check | Observed result |
| --- | --- |
| API probes, OpenAPI, discovery, authentication and submission validation | Passed |
| Real Qwen2.5-0.5B LoRA training, four chat rows, ten small epochs, one GPU | `e2e-00a0f80fd9bd48a1`: `PENDING` → `RUNNING` → `SUCCEEDED` |
| Kueue admission | Independently observed |
| Live log snapshot and SSE | Nonempty logs received; reconnect succeeded |
| Requested one-second stream expiry | Parsed `stream_expired`, closed after 1.001 seconds; job remained `RUNNING` |
| Requested three-second stream expiry | Parsed `stream_expired`, closed after 3.002 seconds with two log events; job remained `RUNNING` |
| Saved output | Two LoRA checkpoints validated, including safetensors structure |
| Terminal deletion | HTTP 204, eventual 404, owned GPU/resource cleanup; saved checkpoints remained readable |
| Pending deletion | `e2e-50f9e77ab2b842e8`: HTTP 204, eventual 404, owned resource cleanup |
| Running deletion | `e2e-013bd39aa4034959`: HTTP 204, eventual 404, owned GPU/resource cleanup |

The service retained the default 300-second stream limit; the live run measured
shorter requested lifetimes, not a full five-minute connection. Skipped cases
were the opt-in four-algorithm matrix and its duplicate cancellation cases,
remote-storage fixtures, prepared missing-submission and invalid-cluster fixtures,
a caller-forbidden project, and two constraints absent from the runtime schema.
Discovery of seven algorithms does not establish that all seven can execute.

A supplemental log request ran **after** foreground deletion had been accepted
at 14:02:41 UTC. It returned 503 at 14:02:57 while teardown was removing the Ray
cluster; the job reached 404 at 14:03:17. Live logging passed before deletion.
Ray-backed logs are not durable after the cluster is removed.

## Repository checks

All **1,020 unit tests**, including **36 Helm tests**, passed. Ruff lint and
format checks, mypy, Helm lint/template checks and OpenShift server-side dry-run
validation passed. No application code changed after the published image build;
the subsequent changes are chart, tests, documentation and this evidence.

[Machine-readable deployment and test evidence](pr14-customer-install.json)
records digests, the empty Helm overrides, runtime identity, timeout timings,
checkpoint validation, deletion results and exact skip reasons.
