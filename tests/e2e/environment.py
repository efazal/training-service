"""Create only isolated E2E fixtures; never reconcile shared DSC/queue policy."""

import hashlib
import json
import os
import ssl
import time
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from tests.e2e.registry import PullSecret, kubeconfig_client

FIXTURES = Path(__file__).with_name("fixtures")
EXAMPLES = Path(__file__).with_name("examples")
RUN_LABEL = "training.rhoai.io/e2e-run"


def namespace_manifest(name, run_id):
    manifest = yaml.safe_load((FIXTURES / "namespace.yaml").read_text())
    manifest["metadata"]["name"] = name
    manifest["metadata"]["labels"][RUN_LABEL] = run_id
    return manifest


def cluster_preflight(args, command, values):
    """Read-only checks before image upload or creating a namespace."""
    if command(
        ["oc", "get", "namespace", args.namespace, "--ignore-not-found", "-o", "name"]
    ).strip():
        raise ValueError("Test namespace already exists; refusing to adopt or modify it")
    # Installation method and queue health are not global suite prerequisites.
    # The queued lifecycle verifies actual Kueue APIs/policy before its POST.
    args.cluster_queue = args.cluster_queue or "default"
    if not args.storage_class:
        raise ValueError(
            "Specify --storage-class with a ReadWriteMany-capable class for test fixtures"
        )
    command(["oc", "get", "storageclass", args.storage_class, "-o", "name"])
    configured = {e["name"]: e.get("value") for e in values.get("env", [])}
    defaults = yaml.safe_load(
        (FIXTURES.parents[2] / "charts/training-service/values.yaml").read_text()
    )
    default_runtime = defaults["trainingRuntime"]["image"]
    runtime = (values.get("trainingRuntime") or {}).get("image", default_runtime)
    if runtime and runtime != default_runtime and "TRAINING_RAY_IMAGE" in configured:
        raise ValueError("Configure trainingRuntime.image or env TRAINING_RAY_IMAGE, not both")
    runtime = configured.get("TRAINING_RAY_IMAGE", runtime)
    if not isinstance(runtime, str) or not runtime or "REPLACE" in runtime or "YOUR_" in runtime:
        raise ValueError(
            "Values must specify a tested trainingRuntime.image for the fixture init container"
        )
    args.runtime_image = runtime
    gateway = configured.get("TRAINING_SERVICE_RAY_GATEWAY_URL")
    parsed = urlsplit(gateway or "")
    if (
        not gateway
        or "YOUR_" in gateway
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Values must specify the authenticated HTTPS TRAINING_SERVICE_RAY_GATEWAY_URL"
        )
    if values.get("imagePullSecrets"):
        raise ValueError(
            "Do not reference existing imagePullSecrets in a fresh namespace; "
            "use --use-podman-auth to create a test-only Quay pull secret."
        )
    for resource in ("namespaces", "clusterroles", "clusterrolebindings"):
        if command(["oc", "auth", "can-i", "create", resource]).strip() != "yes":
            raise ValueError(f"Deployment identity needs create access to {resource}")
    if not args.keep_resources:
        for resource in ("namespaces", "clusterroles", "clusterrolebindings"):
            if command(["oc", "auth", "can-i", "delete", resource]).strip() != "yes":
                raise ValueError(
                    f"Deployment identity needs delete access to {resource} for cleanup"
                )
        for resource in ("clusterroles", "clusterrolebindings"):
            if command(["oc", "auth", "can-i", "list", resource]).strip() != "yes":
                raise ValueError(
                    f"Deployment identity needs list access to {resource} to verify cleanup"
                )


def public_certificates(pem, source):
    """Validate public certificate material before saving or copying it to a ConfigMap."""
    if not isinstance(pem, str) or not pem.strip():
        raise ValueError(f"{source} contains no public certificates")
    if "PRIVATE KEY" in pem:
        raise ValueError(f"{source} must contain public certificates only, never private keys")
    try:
        ssl.create_default_context().load_verify_locations(cadata=pem)
    except (ssl.SSLError, ValueError):
        raise ValueError(f"{source} does not contain a valid PEM certificate bundle") from None
    return pem


def prepare_tls(args, command, results, values):
    """Use authenticated ConfigMap reads, never Secrets or unverified peer certificates."""
    configured = {e["name"]: e.get("value") for e in values.get("env", [])}
    ca_file_configured = configured.get("TRAINING_SERVICE_RAY_CA_FILE")
    if ca_file_configured or args.ray_ca_bundle:
        volumes = {
            v["name"]
            for v in values.get("volumes", [])
            if v.get("configMap", {}).get("name") == "training-service-ray-ca"
        }
        if not ca_file_configured or not any(
            m.get("name") in volumes
            and m.get("readOnly") is True
            and ca_file_configured == m.get("mountPath", "").rstrip("/") + "/ca.crt"
            for m in values.get("volumeMounts", [])
        ):
            raise ValueError(
                "Mount training-service-ray-ca read-only at the configured Ray CA directory; "
                "see tests/e2e/examples/values.example.yaml"
            )
    route_override = os.getenv("TRAINING_SERVICE_CA_BUNDLE")
    ray_override = args.ray_ca_bundle
    route_pem = (
        public_certificates(Path(route_override).read_text(), "TRAINING_SERVICE_CA_BUNDLE")
        if route_override
        else None
    )
    ray_pem = (
        public_certificates(args.ray_ca_bundle.read_text(), "--ray-ca-bundle")
        if args.ray_ca_bundle
        else None
    )
    source = "openshift-config-managed/default-ingress-cert"
    discovered = None
    discovered_file = None
    if not route_override or (ca_file_configured and not args.ray_ca_bundle):
        try:
            configmap = json.loads(
                command(
                    [
                        "oc",
                        "get",
                        "configmap",
                        "default-ingress-cert",
                        "-n",
                        "openshift-config-managed",
                        "-o",
                        "json",
                    ]
                )
            )
        except (RuntimeError, ValueError):
            raise ValueError(
                f"Cannot discover the public ingress CA from ConfigMap {source}. "
                "Grant get access, or set TRAINING_SERVICE_CA_BUNDLE and (if needed) "
                "--ray-ca-bundle. TLS verification is never disabled."
            ) from None
        discovered = public_certificates(configmap.get("data", {}).get("ca-bundle.crt"), source)
        discovered_file = results / "ingress-ca.pem"
        discovered_file.write_text(discovered)
    args.ingress_ca_bundle = Path(route_override) if route_override else discovered_file
    if ca_file_configured and not args.ray_ca_bundle:
        args.ray_ca_bundle = discovered_file
        ray_pem = discovered
    ray_source = "system trust"
    if ray_override:
        ray_source = "--ray-ca-bundle"
    elif ray_pem:
        ray_source = source
    return {
        "ingress_ca_source": "TRAINING_SERVICE_CA_BUNDLE" if route_override else source,
        "ingress_ca_sha256": hashlib.sha256((route_pem or discovered).encode()).hexdigest(),
        "ray_ca_source": ray_source,
    }


class RunEnvironment:
    def __init__(self, args, run_id, results, command):
        self.args, self.run_id, self.results, self.command = args, run_id, results, command
        self.namespace_uid = None
        self.namespace_creation_attempted = False
        self.pod_uid = None
        self.kueue_setup_error = None
        self.pull_secret = PullSecret(args.namespace, run_id) if args.use_podman_auth else None

    def create(self, filename, document):
        path = self.results / filename
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        return json.loads(self.command(["oc", "create", "-f", path, "-o", "json"]))

    def resource(self, kind, name, **spec):
        version = "kueue.x-k8s.io/v1beta1" if kind == "LocalQueue" else "v1"
        return {
            "apiVersion": version,
            "kind": kind,
            "metadata": {
                "name": name,
                "namespace": self.args.namespace,
                "labels": {RUN_LABEL: self.run_id},
            },
            **spec,
        }

    def fixture_group(self, namespace):
        """Use an OpenShift-allocated group, never a hardcoded UID or privileged pod."""
        deadline = time.monotonic() + min(30, self.args.fixture_timeout)
        while True:
            annotations = namespace["metadata"].get("annotations", {})
            allocated = annotations.get(
                "openshift.io/sa.scc.supplemental-groups"
            ) or annotations.get("openshift.io/sa.scc.uid-range", "")
            if allocated:
                try:
                    first, size = allocated.split(",")[0].split("/")
                    if int(first) > 0 and int(size) > 0:
                        return int(first)
                except ValueError:
                    pass
                raise RuntimeError("Namespace has an invalid allocated filesystem group range")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Namespace has no OpenShift-allocated filesystem group; "
                    "ask an administrator to check namespace security allocation"
                )
            time.sleep(1)
            namespace = json.loads(
                self.command(
                    ["oc", "get", "namespace", self.args.namespace, "-o", "json"], timeout=15
                )
            )

    def wait_for_fixtures(self):
        """Fail on terminated setup containers instead of waiting out the readiness deadline."""
        deadline = time.monotonic() + self.args.fixture_timeout
        previous = None
        last_progress = 0
        while True:
            pod = json.loads(
                self.command(
                    [
                        "oc",
                        "get",
                        "pod/e2e-output-verifier",
                        "-n",
                        self.args.namespace,
                        "-o",
                        "json",
                    ],
                    timeout=15,
                )
            )
            if pod["metadata"]["uid"] != self.pod_uid:
                raise RuntimeError("Fixture pod ownership changed while waiting for readiness")
            status = pod.get("status", {})
            containers = status.get("initContainerStatuses", []) + status.get(
                "containerStatuses", []
            )
            progress = [status.get("phase", "Pending")]
            for container in containers:
                state = container.get("state", {})
                terminated = state.get("terminated", {})
                if terminated.get("exitCode", 0) != 0:
                    raise RuntimeError(
                        f"Fixture container '{container['name']}' exited with code "
                        f"{terminated['exitCode']} in {self.args.namespace}/e2e-output-verifier. "
                        f"Inspect: oc logs -n {self.args.namespace} e2e-output-verifier "
                        f"-c {container['name']}. "
                        "Check PVC write permissions/fsGroup if preparation failed."
                    )
                state_name = state.get("waiting", {}).get("reason") or next(iter(state), "waiting")
                progress.append(f"{container['name']}: {state_name}")
            if status.get("phase") in {"Failed", "Succeeded"}:
                raise RuntimeError(f"Fixture pod ended as {status['phase']} before becoming Ready")
            if any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for c in status.get("conditions", [])
            ):
                print("Fixture pod Ready; model and dataset prepared", flush=True)
                return
            summary = ", ".join(progress)
            now = time.monotonic()
            if summary != previous or now - last_progress >= 30:
                print(f"Waiting for fixtures: {summary}", flush=True)
                previous, last_progress = summary, now
            remaining = deadline - now
            if remaining <= 0:
                raise RuntimeError(
                    f"Fixture readiness deadline exceeded ({summary}); inspect pod events/PVC "
                    f"in namespace '{self.args.namespace}'"
                )
            time.sleep(min(2, remaining))

    def provision(self):
        args = self.args
        self.namespace_creation_attempted = True
        namespace = self.create("namespace.yaml", namespace_manifest(args.namespace, self.run_id))
        self.namespace_uid = namespace["metadata"]["uid"]
        if self.pull_secret:
            self.pull_secret.create(self.namespace_uid)
        group = self.fixture_group(namespace)
        try:
            self.create(
                "localqueue.yaml",
                self.resource("LocalQueue", "e2e", spec={"clusterQueue": args.cluster_queue}),
            )
        except RuntimeError:
            self.kueue_setup_error = (
                f"LocalQueue '{args.namespace}/e2e' creation could not be confirmed for "
                f"ClusterQueue '{args.cluster_queue}'; check Kueue APIs, create permissions "
                "and admission webhooks. The queued test will diagnose the actual queue; "
                "independent API tests will continue."
            )
            print(self.kueue_setup_error, flush=True)
        self.create(
            "storage.yaml",
            self.resource(
                "PersistentVolumeClaim",
                "e2e-storage",
                spec={
                    "accessModes": ["ReadWriteMany"],
                    "storageClassName": args.storage_class,
                    "resources": {"requests": {"storage": "5Gi"}},
                },
            ),
        )
        self.create(
            "fixture-source.yaml",
            self.resource(
                "ConfigMap",
                "e2e-fixture-source",
                data={
                    "prepare.py": (FIXTURES / "prepare.py").read_text(),
                },
            ),
        )
        if args.ray_ca_bundle:
            self.create(
                "ray-ca.yaml",
                self.resource(
                    "ConfigMap",
                    "training-service-ray-ca",
                    data={
                        "ca.crt": args.ray_ca_bundle.read_text(),
                    },
                ),
            )
        security = {
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
        }
        resources = {
            "requests": {"cpu": "100m", "memory": "256Mi"},
            "limits": {"cpu": "1", "memory": "2Gi"},
        }
        # No queue name on setup: Kueue's RayJob integration manages the training job.
        pod = self.resource(
            "Pod",
            "e2e-output-verifier",
            spec={
                "restartPolicy": "Never",
                "automountServiceAccountToken": False,
                "securityContext": {
                    "runAsNonRoot": True,
                    "fsGroup": group,
                    "fsGroupChangePolicy": "OnRootMismatch",
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "initContainers": [
                    {
                        "name": "prepare",
                        "image": args.runtime_image,
                        "command": ["python", "/fixture/prepare.py"],
                        "securityContext": security,
                        "resources": resources,
                        "env": [
                            {"name": "HF_HUB_DISABLE_IMPLICIT_TOKEN", "value": "1"},
                            {"name": "HF_HUB_DISABLE_XET", "value": "1"},
                            {"name": "HF_HOME", "value": "/tmp/hf"},
                        ],
                        "volumeMounts": [
                            {"name": "storage", "mountPath": "/data"},
                            {"name": "source", "mountPath": "/fixture", "readOnly": True},
                            {"name": "tmp", "mountPath": "/tmp"},
                        ],
                    }
                ],
                "containers": [
                    {
                        "name": "verifier",
                        "image": args.runtime_image,
                        "command": ["python", "-c", "import time; time.sleep(86400)"],
                        "securityContext": security,
                        "resources": resources,
                        "volumeMounts": [
                            {"name": "storage", "mountPath": "/data", "readOnly": True},
                            {"name": "tmp", "mountPath": "/tmp"},
                        ],
                    }
                ],
                "volumes": [
                    {"name": "storage", "persistentVolumeClaim": {"claimName": "e2e-storage"}},
                    {"name": "source", "configMap": {"name": "e2e-fixture-source"}},
                    {"name": "tmp", "emptyDir": {}},
                ],
            },
        )
        if self.pull_secret:
            pod["spec"]["imagePullSecrets"] = [{"name": self.pull_secret.name}]
        self.pod_uid = self.create("fixtures.yaml", pod)["metadata"]["uid"]
        self.wait_for_fixtures()
        payload = json.loads((EXAMPLES / "job.example.json").read_text())
        payload.update(queue_name="e2e")
        payload["model"]["uri"] = "pvc://e2e-storage/models/qwen/"
        payload["dataset"]["uri"] = "pvc://e2e-storage/datasets/tiny-chat.jsonl"
        payload["output"]["destination_uri"] = "pvc://e2e-storage/outputs/"
        payload["resources"]["cpus_per_node"] = "4"
        jobfile = self.results / "workload.json"
        jobfile.write_text(json.dumps(payload, indent=2) + "\n")
        args.job_file = jobfile
        args.artifact_pod = "e2e-output-verifier"
        args.artifact_container = "verifier"

    def cleanup(self, release, deployed, evidence):
        """Remove only our fresh namespace, after uninstalling cluster-scoped Helm resources."""
        evidence["namespace_cleanup"] = "not created"
        if not self.namespace_uid and not self.namespace_creation_attempted:
            return
        print(f"Cleaning up test resources in {self.args.namespace}", flush=True)
        try:
            with kubeconfig_client() as api:
                core = client.CoreV1Api(api)
                try:
                    current = core.read_namespace(self.args.namespace, _request_timeout=15)
                except ApiException as error:
                    if error.status != 404:
                        raise
                    if deployed:
                        raise RuntimeError(
                            "Namespace missing; inspect Helm cluster-scoped resources"
                        ) from None
                    evidence["namespace_cleanup"] = "already absent"
                    return
                if (self.namespace_uid and current.metadata.uid != self.namespace_uid) or (
                    current.metadata.labels or {}
                ).get(RUN_LABEL) != self.run_id:
                    raise RuntimeError("Namespace ownership changed; refusing resource cleanup")
                # Recover an ambiguous namespace POST only when its unique run label matches.
                self.namespace_uid = current.metadata.uid
                if deployed:
                    self.command(
                        [
                            "helm",
                            "uninstall",
                            release,
                            "-n",
                            self.args.namespace,
                            "--ignore-not-found",
                            "--wait",
                            "--timeout",
                            f"{self.args.cleanup_timeout}s",
                        ],
                        timeout=self.args.cleanup_timeout + 30,
                    )
                    remaining_rbac = json.loads(
                        self.command(
                            [
                                "oc",
                                "get",
                                "clusterroles,clusterrolebindings",
                                "-l",
                                "app.kubernetes.io/instance=" + release,
                                "-o",
                                "json",
                            ]
                        )
                    )
                    if remaining_rbac.get("items"):
                        raise RuntimeError(
                            "Helm cluster-scoped RBAC remains; inspect the test release"
                        )
                    evidence["deployment_cleanup"] = "removed"
                core.delete_namespace(
                    self.args.namespace,
                    body=client.V1DeleteOptions(
                        preconditions=client.V1Preconditions(uid=self.namespace_uid)
                    ),
                    _request_timeout=15,
                )
                deadline = time.monotonic() + self.args.cleanup_timeout
                while True:
                    try:
                        remaining = core.read_namespace(self.args.namespace, _request_timeout=15)
                    except ApiException as error:
                        if error.status != 404:
                            raise
                        evidence["namespace_cleanup"] = "removed"
                        evidence["fixture_cleanup"] = "removed with namespace (including test PVC)"
                        return
                    if remaining.metadata.uid != self.namespace_uid:
                        raise RuntimeError("Namespace was replaced during cleanup")
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Namespace deletion deadline exceeded; inspect finalizers"
                        )
                    time.sleep(min(2, max(0, deadline - time.monotonic())))
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError(
                "Resource cleanup failed; inspect the test namespace/release"
            ) from None


def terminal_summary(evidence, results):
    def report(filename):
        path = results / filename
        return json.loads(path.read_text()) if path.is_file() else {}

    tests, lifecycle = report("tests.json"), report("lifecycle.json")
    if not lifecycle:
        lifecycle = report("lifecycle-lora_sft.json")
    checkpoints = "verified" if isinstance(lifecycle.get("artifacts"), dict) else "not verified"
    print("\nE2E SUMMARY", flush=True)
    print(f"Result: {evidence['outcome'].upper()} | namespace: {evidence['namespace']}")
    print(
        f"Tests: {tests.get('passed', 'not recorded')} passed, "
        f"{tests.get('failed', 'not recorded')} failed, "
        f"{tests.get('skipped', 'not recorded')} skipped"
    )
    print(
        f"Training SUCCEEDED: {lifecycle.get('succeeded', 'not verified')} | "
        f"Kueue admitted: {lifecycle.get('queue_admitted', 'not verified')}"
    )
    print(
        f"GPU cleanup verified: {lifecycle.get('gpu_cleanup', 'not verified')} | "
        f"checkpoints: {checkpoints}"
    )
    if evidence.get("algorithm_matrix"):
        for name in ("sft", "osft", "lora_sft", "lora_grpo"):
            result = report(f"lifecycle-{name}.json")
            print(f"{name}: {'SUCCEEDED' if result.get('succeeded') else 'not verified'}")
    print(f"Deployment cleanup: {evidence.get('deployment_cleanup', 'not deployed')}")
    print(f"Namespace cleanup: {evidence.get('namespace_cleanup', 'not created')}")
    if evidence.get("cleanup_error"):
        print(f"Cleanup error: {evidence['cleanup_error']}")
    if "registry_secret_cleanup" in evidence:
        print(f"Registry credential cleanup: {evidence['registry_secret_cleanup']}")
    if evidence.get("kueue_setup_error"):
        print(f"Kueue setup: {evidence['kueue_setup_error']}")
    for failure in tests.get("failures", []):
        print(f"Failed {failure['test']}: {failure['reason']}")
    print(f"Evidence/manifests: {results}")
    if evidence.get("namespace_cleanup") == "removed":
        print("Test namespace and storage deleted; local reports retained. Quay image unchanged.")
