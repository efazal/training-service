"""Read-only cluster observer, separate from the HTTP caller's credentials."""

import json
import re
import subprocess
import time
from pathlib import PurePosixPath

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from tests.e2e.support import pvc_path, selector_matches

TIMEOUT = 15


class ObserverError(ValueError):
    """A deliberately credential-free diagnostic safe to show in test reports."""


def failure_reason(error):
    if isinstance(error, ObserverError):
        return str(error)
    if isinstance(error, ApiException):
        return (
            f"Kubernetes observer returned HTTP {error.status}; "
            "check observer permissions/connectivity"
        )
    return "Check kubeconfig, connectivity, observer permissions and fixtures (raw error withheld)"


def conditions_summary(conditions):
    # Never include arbitrary controller messages or API response bodies in evidence.
    fields = []
    for condition in conditions:
        parts = [condition.get(key, "Unknown") for key in ("type", "status", "reason")]
        if all(isinstance(p, str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", p) for p in parts):
            fields.append(f"{parts[0]}={parts[1]} ({parts[2]})")
    return "; ".join(fields) or "no conditions reported"


def pages(fetch):
    token = None
    seen = set()
    for _ in range(100):
        page = fetch(limit=100, _continue=token, _request_timeout=TIMEOUT)
        if not isinstance(page, dict):
            with client.ApiClient() as serializer:
                page = serializer.sanitize_for_serialization(page)
        yield from page.get("items", [])
        token = page.get("metadata", {}).get("continue")
        if not token:
            return
        if token in seen:
            break
        seen.add(token)
    raise ValueError("Observer inventory pagination did not finish")


def owned_by(resource, uids):
    return any(
        ref.get("uid") in uids and ref.get("controller") is True
        for ref in resource.get("metadata", {}).get("ownerReferences", [])
    )


class Observer:
    def __init__(self, project):
        self.project = project
        settings = client.Configuration()
        config.load_kube_config(client_configuration=settings)
        self.api = client.ApiClient(settings)
        self.core = client.CoreV1Api(self.api)
        self.custom = client.CustomObjectsApi(self.api)
        self.batch = client.BatchV1Api(self.api)
        self.uids = set()
        self.job_uid = None
        self.resources = set()
        self.saw_gpu = False
        self.admitted = False
        self.queue = None
        self.profile = None
        self.preserved = []
        self.queue_status = (
            "no owned Workload observed; check RayJob integration/namespace enrollment"
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.api.close()

    def listing(self, plural, group="ray.io", version="v1"):
        return list(
            pages(
                lambda **kw: self.custom.list_namespaced_custom_object(
                    group, version, self.project, plural, **kw
                )
            )
        )

    def inventory(self):
        clusters = self.listing("rayclusters")
        jobs = list(pages(lambda **kw: self.batch.list_namespaced_job(self.project, **kw)))
        pods = list(pages(lambda **kw: self.core.list_namespaced_pod(self.project, **kw)))
        return (
            [("rayclusters", c) for c in clusters]
            + [("jobs", j) for j in jobs]
            + [("pods", p) for p in pods]
        )

    def preflight(self, payload, profile):
        self.profile = profile
        project = self.core.read_namespace(self.project, _request_timeout=TIMEOUT)
        groups = client.ApisApi(self.api).get_api_versions(_request_timeout=TIMEOUT).groups
        kueue = any(g.name == "kueue.x-k8s.io" for g in groups)
        if profile == "no-kueue" and kueue:
            raise ObserverError(
                "Kueue APIs exist: use a Kueue-absent environment or record unqueued-opt-out"
            )
        if profile == "with-kueue":
            if not kueue:
                raise ObserverError(
                    "Kueue API group is absent; install Kueue and its RayJob integration"
                )
            self.queue = payload["queue_name"]
            self.queue_preflight(project.metadata.labels or {}, payload)
        self.inventory()  # Fail on insufficient observer permissions before POST.
        # Preserve metadata only; never read Secrets or their values.
        for role in ("model", "dataset", "output"):
            uri = payload[role].get("uri", payload[role].get("destination_uri", ""))
            if uri.startswith("pvc://"):
                claim, _ = pvc_path(uri)
                record = self.core.read_namespaced_persistent_volume_claim(
                    claim, self.project, _request_timeout=TIMEOUT
                )
                self.preserved.append((claim, record.metadata.uid))

    def queue_preflight(self, labels, payload):
        context = f"LocalQueue '{self.project}/{self.queue}'"
        try:
            local = self.custom.get_namespaced_custom_object(
                "kueue.x-k8s.io",
                "v1beta1",
                self.project,
                "localqueues",
                self.queue,
                _request_timeout=TIMEOUT,
            )
            context = f"ClusterQueue '{local['spec']['clusterQueue']}'"
            cluster = self.custom.get_cluster_custom_object(
                "kueue.x-k8s.io",
                "v1beta1",
                "clusterqueues",
                local["spec"]["clusterQueue"],
                _request_timeout=TIMEOUT,
            )
            if not any(
                c.get("type") == "Active" and c.get("status") == "True"
                for c in cluster.get("status", {}).get("conditions", [])
            ):
                raise ObserverError(
                    f"{context} is not Active: "
                    f"{conditions_summary(cluster.get('status', {}).get('conditions', []))}; "
                    "check ResourceFlavors, admission checks and quota"
                )
            if not selector_matches(labels, cluster["spec"].get("namespaceSelector", {})):
                raise ObserverError(
                    f"{context} does not admit namespace '{self.project}': "
                    "namespaceSelector mismatch"
                )
            covered = {
                r
                for group in cluster["spec"].get("resourceGroups", [])
                for r in group.get("coveredResources", [])
            }
            gpu = payload.get("resources", {}).get("gpu_type", "nvidia.com/gpu")
            missing = {"cpu", "memory", gpu} - covered
            if missing:
                raise ObserverError(
                    f"{context} does not cover requested resources: {', '.join(sorted(missing))}"
                )
            context = f"Kueue Workloads in namespace '{self.project}'"
            self.listing("workloads", "kueue.x-k8s.io", "v1beta1")
        except ApiException as error:
            reason = {
                401: "observer credentials were rejected",
                403: "observer permission denied",
                404: "not found (check Kueue installation and queue names)",
            }.get(
                error.status, "Kubernetes request failed; check API connectivity/controller health"
            )
            raise ObserverError(f"{context}: {reason} (HTTP {error.status})") from None

    def queue_diagnostic(self):
        return (
            f"Kueue LocalQueue '{self.project}/{self.queue}': {self.queue_status}"
            if self.queue
            else ""
        )

    def observe(self, job_id):
        try:
            job = self.custom.get_namespaced_custom_object(
                "ray.io", "v1", self.project, "rayjobs", job_id, _request_timeout=TIMEOUT
            )
        except ApiException as error:
            if error.status != 404:
                raise
            job = None
        if job:
            uid = job["metadata"]["uid"]
            if self.job_uid and uid != self.job_uid:
                raise ValueError("Observed job UID changed; refusing to follow a replacement")
            self.job_uid = uid
            self.uids.add(uid)
            selected_queue = job["metadata"].get("labels", {}).get("kueue.x-k8s.io/queue-name")
            if self.profile in {"no-kueue", "unqueued-opt-out"} and selected_queue:
                raise ValueError(
                    "SDK selected a default queue; this is not unqueued execution proof"
                )
            if self.queue and selected_queue != self.queue:
                raise ValueError("RayJob did not preserve the selected LocalQueue")
        inventory = self.inventory()
        # Transitive ownership: RayJob -> RayCluster/batch Job -> pods.
        for _ in range(4):
            for kind, item in inventory:
                if owned_by(item, self.uids):
                    metadata = item["metadata"]
                    self.uids.add(metadata["uid"])
                    self.resources.add((kind, metadata["name"], metadata["uid"]))
                    if kind == "pods" and item.get("status", {}).get("phase") == "Running":
                        for container in item.get("spec", {}).get("containers", []):
                            requests = container.get("resources", {}).get("requests", {})
                            if any(
                                k.endswith("/gpu") and float(v) > 0 for k, v in requests.items()
                            ):
                                self.saw_gpu = True
        if self.queue:
            for item in self.listing("workloads", "kueue.x-k8s.io", "v1beta1"):
                if owned_by(item, self.uids):
                    self.uids.add(item["metadata"]["uid"])
                    self.resources.add(
                        ("workloads", item["metadata"]["name"], item["metadata"]["uid"])
                    )
                    if item.get("spec", {}).get("queueName") != self.queue:
                        raise ValueError("Workload did not preserve the selected LocalQueue")
                    self.admitted |= any(
                        c["type"] == "Admitted" and c["status"] == "True"
                        for c in item.get("status", {}).get("conditions", [])
                    )
                    self.queue_status = conditions_summary(
                        item.get("status", {}).get("conditions", [])
                    )
        return inventory

    def verify_cleanup(self, job_id, timeout):
        deadline = time.monotonic() + timeout
        while True:
            inventory = self.observe(job_id)
            remaining = [
                (kind, item) for kind, item in inventory if item["metadata"]["uid"] in self.uids
            ]
            try:
                self.custom.get_namespaced_custom_object(
                    "ray.io", "v1", self.project, "rayjobs", job_id, _request_timeout=TIMEOUT
                )
                job_gone = False
            except ApiException as error:
                if error.status != 404:
                    raise
                job_gone = True
            workloads = self.listing("workloads", "kueue.x-k8s.io", "v1beta1") if self.queue else []
            pending_workloads = any(w["metadata"]["uid"] in self.uids for w in workloads)
            if job_gone and not remaining and not pending_workloads:
                break
            if time.monotonic() >= deadline:
                raise ValueError(
                    "Test-owned RayJob/cluster/pods/Workload remain after cleanup deadline"
                )
            time.sleep(2)
        if not self.job_uid or not self.saw_gpu:
            raise ValueError(
                "No running owned GPU pod was observed; GPU-release proof is incomplete"
            )
        if self.queue and not self.admitted:
            raise ValueError("No owned admitted Kueue Workload was observed")
        for name, uid in self.preserved:
            pvc = self.core.read_namespaced_persistent_volume_claim(
                name, self.project, _request_timeout=TIMEOUT
            )
            if pvc.metadata.uid != uid:
                raise ValueError("A pre-existing PVC was replaced during the run")
        # Check the shared resource API survives, without requiring cluster-wide CRD reads.
        self.listing("rayjobs")

    def artifact_mount(self, pod, container, payload):
        claim, path = pvc_path(payload["output"]["destination_uri"])
        record = self.core.read_namespaced_pod(pod, self.project, _request_timeout=TIMEOUT)
        volumes = {
            v.name
            for v in record.spec.volumes
            if v.persistent_volume_claim and v.persistent_volume_claim.claim_name == claim
        }
        for candidate in record.spec.containers:
            if candidate.name == container:
                for mount in candidate.volume_mounts or []:
                    if (
                        mount.name in volumes
                        and mount.read_only
                        and not mount.sub_path
                        and not mount.sub_path_expr
                    ):
                        return str(PurePosixPath(mount.mount_path) / path)
        raise ValueError(
            "Verifier must mount the complete output PVC read-only in the selected container"
        )

    def artifacts(self, pod, container, payload):
        path = self.artifact_mount(pod, container, payload)
        # Validate LoRA config and safetensors structure/data bounds, not just directory existence.
        script = """
import json, pathlib, struct, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
valid = 0
for config in root.rglob("adapter_config.json"):
    assert config.resolve().is_relative_to(root)
    cfg = json.loads(config.read_text())
    assert cfg.get("peft_type") == "LORA" and cfg.get("r", 0) > 0
    weights = config.with_name("adapter_model.safetensors")
    assert weights.resolve(strict=True).is_relative_to(root)
    size = weights.stat().st_size
    with weights.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        assert 0 < header_size < min(size - 8, 16 * 1024 * 1024)
        header = json.loads(stream.read(header_size))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    assert tensors
    for tensor in tensors:
        start, end = tensor["data_offsets"]
        assert 0 <= start < end <= size - 8 - header_size
        assert tensor["dtype"] and tensor["shape"]
    valid += 1
assert valid, "No usable LoRA adapter checkpoint found"
print(json.dumps({"validated_lora_checkpoints": valid}))
"""
        try:
            result = subprocess.run(
                [
                    "oc",
                    "exec",
                    "-n",
                    self.project,
                    pod,
                    "-c",
                    container,
                    "--",
                    "python",
                    "-c",
                    script,
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            data = json.loads(result.stdout) if result.returncode == 0 else {}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            data = {}
        if (
            not isinstance(data.get("validated_lora_checkpoints"), int)
            or data["validated_lora_checkpoints"] < 1
        ):
            raise ValueError("Output verifier failed: inspect the retained per-run LoRA artifacts")
        return data
