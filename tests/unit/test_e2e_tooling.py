"""Credential-free runner/observer safety tests, not substitutes for live E2E."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from kubernetes.client.exceptions import ApiException
from tests.e2e import deploy
from tests.e2e.conftest import LiveAPI
from tests.e2e.environment import (
    RUN_LABEL,
    RunEnvironment,
    cluster_preflight,
    namespace_manifest,
    terminal_summary,
)
from tests.e2e.observer import Observer, ObserverError, failure_reason, owned_by, pages
from tests.e2e.support import pvc_path, selector_matches, workload
from tests.e2e.test_training_jobs import follow_logs


@pytest.fixture
def jobfile(tmp_path):
    payload = json.loads((deploy.ROOT / "tests/e2e/job.example.json").read_text())
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload))
    return path


def test_profiles_require_omitted_queue_and_explicit_selection(jobfile):
    with pytest.raises(ValueError, match="explicit"):
        workload(jobfile, None)
    with pytest.raises(ValueError, match="non-empty"):
        workload(jobfile, "with-kueue")
    payload = workload(jobfile, "no-kueue")
    payload["queue_name"] = ""
    jobfile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="omit"):
        workload(jobfile, "no-kueue")
    payload["queue_name"] = "prepared-local-queue"
    jobfile.write_text(json.dumps(payload))
    assert workload(jobfile, "with-kueue")["queue_name"] == "prepared-local-queue"


@pytest.mark.parametrize(
    "uri",
    [
        "pvc://output/../other",
        "pvc://output/%2e%2e/other",
        "pvc://output/path?token=private",
        "pvc://user:password@output/path",
        "pvc://output/a//b",
        "s3://output/bucket/prefix",
    ],
)
def test_artifact_paths_reject_unsafe_references(uri):
    with pytest.raises(ValueError):
        pvc_path(uri)


def test_workloads_reject_embedded_credentials(jobfile):
    payload = json.loads(jobfile.read_text())
    payload["model"]["uri"] = "hf://token@example/model"
    jobfile.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="credentials"):
        workload(jobfile, "no-kueue")


def test_pages_follow_all_continuations_and_reject_loops():
    fetch = Mock(
        side_effect=[
            {"items": [{"id": 1}], "metadata": {"continue": "next"}},
            {"items": [{"id": 2}], "metadata": {}},
        ]
    )
    assert list(pages(fetch)) == [{"id": 1}, {"id": 2}]
    assert fetch.call_args.kwargs["_continue"] == "next"
    fetch = Mock(return_value={"items": [], "metadata": {"continue": "repeated"}})
    with pytest.raises(ValueError, match="pagination"):
        list(pages(fetch))


def test_observer_requires_controller_owner_uid():
    item = {"metadata": {"ownerReferences": [{"uid": "owned", "controller": True}]}}
    assert owned_by(item, {"owned"})
    assert not owned_by(item, {"different"})
    item["metadata"]["ownerReferences"][0]["controller"] = False
    assert not owned_by(item, {"owned"})


def test_observer_refuses_job_uid_replacement():
    observer = object.__new__(Observer)
    observer.project = "test-project"
    observer.job_uid = "old-uid"
    observer.custom = Mock()
    observer.custom.get_namespaced_custom_object.return_value = {"metadata": {"uid": "replacement"}}
    with pytest.raises(ValueError, match="UID changed"):
        observer.observe("test-job")


def test_observer_does_not_claim_gpu_cleanup_without_observation():
    observer = object.__new__(Observer)
    observer.project = "test-project"
    observer.queue = None
    observer.job_uid = "test-uid"
    observer.saw_gpu = False
    observer.custom = Mock()
    observer.custom.get_namespaced_custom_object.side_effect = ApiException(status=404)
    observer.observe = Mock(return_value=[])
    with pytest.raises(ValueError, match="GPU-release proof is incomplete"):
        observer.verify_cleanup("test-job", 1)


@pytest.mark.parametrize(
    "cmd", [["helm", "install", "test"], [sys.executable, "-m", "pytest", "tests/e2e"]]
)
def test_subprocess_errors_withhold_raw_output(monkeypatch, cmd):
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                ["helm"],
                1,
                stdout="private-token",
                stderr="private-secret",
            )
        ),
    )
    with pytest.raises(RuntimeError) as error:
        deploy.command(cmd)
    assert "private" not in str(error.value)
    assert "exit 1" in str(error.value)


@pytest.mark.parametrize(
    "failure,keep,missing_evidence",
    [(False, False, False), (False, True, False), (True, False, False), (False, False, True)],
)
@pytest.mark.parametrize("reuse", [False, True])
def test_runner_build_push_digest_install_test_cleanup(
    monkeypatch, tmp_path, failure, keep, missing_evidence, reuse
):
    values = tmp_path / "values.yaml"
    values.write_text("env: []\n")
    args = deploy.arguments(
        [
            "--image-repository",
            "quay.io/test/training-service",
            "--namespace",
            "test-project",
            "--values",
            str(values),
            "--results-dir",
            str(tmp_path / "results"),
            *(["--keep-deployment"] if keep else []),
            *(
                ["--reuse-image", "quay.io/test/training-service@sha256:" + "a" * 64]
                if reuse
                else []
            ),
        ]
    )
    monkeypatch.setattr(deploy, "preflight", Mock())
    args.ingress_ca_bundle = tmp_path / "generated-ingress-ca.pem"
    monkeypatch.setattr(deploy, "prepare_tls", Mock(return_value={"ingress_ca_source": "test"}))
    monkeypatch.setattr(deploy, "wait_ready", Mock())
    environment = Mock(namespace_uid="test-namespace-uid", kueue_setup_error=None)

    def provision():
        args.job_file = tmp_path / "generated-workload.json"
        args.artifact_pod = "e2e-output-verifier"
        args.artifact_container = "verifier"

    environment.provision.side_effect = provision
    monkeypatch.setattr(deploy, "RunEnvironment", Mock(return_value=environment))
    commands = []
    digest = "sha256:" + "a" * 64

    def run(command, **kwargs):
        command = [str(value) for value in command]
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "b" * 40
        if command[:2] == ["helm", "list"]:
            return "[]"
        if command[:2] == ["podman", "push"]:
            Path(command[command.index("--digestfile") + 1]).write_text(digest)
        if command[:3] == ["oc", "get", "route"]:
            return json.dumps({"spec": {"host": "test.example.com"}})
        if command[:3] == ["oc", "get", "pods"]:
            return '{"items": []}'
        if command[:2] == ["oc", "exec"]:
            return json.dumps(
                {"training-service": "0.1.0", "ray": "2.55.1", "codeflare-sdk": "0.38.2"}
            )
        if command[:2] == ["oc", "version"]:
            return "{}"
        if "pytest" in command:
            assert kwargs["env"]["TRAINING_SERVICE_CA_BUNDLE"] == str(args.ingress_ca_bundle)
            if failure:
                raise RuntimeError("live tests failed")
            if missing_evidence:
                return ""
            results = Path(kwargs["env"]["TRAINING_SERVICE_EVIDENCE_DIR"])
            (results / "lifecycle.json").write_text(
                json.dumps(
                    {
                        "succeeded": True,
                        "queue_admitted": True,
                        "gpu_cleanup": True,
                        "snapshot_logs": True,
                        "sse": True,
                        "artifacts": {"adapters": 1},
                    }
                )
            )
        return ""

    monkeypatch.setattr(deploy, "command", run)
    failed = failure or missing_evidence
    if failed:
        with pytest.raises(RuntimeError, match="live tests failed|evidence is missing"):
            deploy.run(args)
    else:
        deploy.run(args)
    builds = [c for c in commands if c[:2] == ["podman", "build"]]
    assert bool(builds) != reuse
    assert any(c[:2] == ["podman", "push"] for c in commands) != reuse
    if builds:
        assert "Containerfile" in builds[0] and "Dockerfile" not in builds[0]
    install = next(c for c in commands if c[:2] == ["helm", "install"])
    listing = next(c for c in commands if c[:2] == ["helm", "list"])
    assert "--all" not in listing  # Removed in Helm 4; explicit status flags also work in Helm 3.
    assert {
        "--deployed",
        "--failed",
        "--pending",
        "--superseded",
        "--uninstalled",
        "--uninstalling",
    } <= set(listing)
    prefix = "image.tag=reused" if reuse else "image.tag=e2e-"
    assert any(c.startswith(prefix) and c.endswith("@" + digest) for c in install)
    assert install[2].startswith("training-e2e-") and "upgrade" not in install
    invocation = next(c for c in commands if "pytest" in c)
    assert invocation[invocation.index("--queue-profile") + 1] == "with-kueue"
    assert "--maxfail=0" in invocation
    assert {"--job-file", "--artifact-pod", "--observe-cluster", "--check-stream"} <= set(
        invocation
    )
    environment.provision.assert_called_once()
    assert environment.cleanup_setup_pod.called == (not failed and not keep)
    assert any(c[:2] == ["helm", "uninstall"] for c in commands) == (not failed and not keep)
    report = json.loads(next((tmp_path / "results").glob("*/run.json")).read_text())
    assert report["outcome"] == ("failed" if failed else "passed")
    assert report["image"].endswith("@" + digest)
    assert report["image_reused"] == reuse


@pytest.fixture
def prepared_cluster(tmp_path):
    values = {
        "env": [
            {"name": "TRAINING_RAY_IMAGE", "value": "quay.io/test/ray:tested"},
            {"name": "TRAINING_SERVICE_RAY_GATEWAY_URL", "value": "https://ray.example.com"},
        ]
    }
    args = deploy.arguments(
        [
            "--image-repository",
            "quay.io/test/training-service",
            "--namespace",
            "new-test-project",
            "--values",
            str(tmp_path / "values.yaml"),
            "--storage-class",
            "shared-rwx",
        ]
    )
    dsc = {
        "metadata": {"name": "default-dsc"},
        "spec": {
            "components": {
                "ray": {"managementState": "Managed"},
                "kueue": {"managementState": "Managed", "defaultClusterQueueName": "default"},
            }
        },
        "status": {
            "conditions": [
                {"type": "RayReady", "status": "True"},
                {"type": "KueueReady", "status": "True"},
            ]
        },
    }
    queue = {
        "spec": {
            "namespaceSelector": {"matchLabels": {"kueue.openshift.io/managed": "true"}},
            "resourceGroups": [{"coveredResources": ["cpu", "memory", "nvidia.com/gpu"]}],
        },
        "status": {"conditions": [{"type": "Active", "status": "True"}]},
    }

    def query(cmd, **kwargs):
        if cmd[:3] == ["oc", "get", "datascienceclusters"]:
            return json.dumps({"items": [dsc]})
        if cmd[:3] == ["oc", "get", "clusterqueue"]:
            return json.dumps(queue)
        if cmd[:3] == ["oc", "auth", "can-i"]:
            return "yes"
        return ""

    return args, values, dsc, queue, Mock(side_effect=query)


def test_fresh_namespace_manifest_has_kueue_enrollment_and_unique_owner():
    manifest = namespace_manifest("new-project", "unique-run")
    assert manifest["metadata"]["name"] == "new-project"
    assert manifest["metadata"]["labels"]["kueue.openshift.io/managed"] == "true"
    assert manifest["metadata"]["labels"][RUN_LABEL] == "unique-run"


@pytest.mark.parametrize(
    "operator,values,expected",
    [
        ("In", ["true"], True),
        ("In", ["false"], False),
        ("NotIn", ["true"], False),
        ("NotIn", ["false"], True),
        ("Exists", [], True),
        ("DoesNotExist", [], False),
        ("Invalid", [], False),
    ],
)
def test_cluster_queue_namespace_expressions(operator, values, expected):
    assert (
        selector_matches(
            {"managed": "true"},
            {
                "matchExpressions": [
                    {"key": "managed", "operator": operator, "values": values},
                ]
            },
        )
        == expected
    )


def test_cluster_preflight_validates_queued_environment_read_only(prepared_cluster):
    args, values, _, _, command = prepared_cluster
    cluster_preflight(args, command, values)
    assert args.cluster_queue == "default"
    assert args.runtime_image == "quay.io/test/ray:tested"
    assert all(call.args[0][1] in {"get", "auth"} for call in command.call_args_list)


def test_cluster_preflight_accepts_directly_installed_kueue(prepared_cluster):
    args, values, dsc, queue, command = prepared_cluster
    dsc["spec"]["components"]["kueue"]["managementState"] = "Removed"
    dsc["status"]["conditions"][1]["status"] = "False"
    queue["status"]["conditions"][0]["status"] = "False"
    cluster_preflight(args, command, values)
    assert args.cluster_queue == "default"
    assert not any(
        call.args[0][2] in {"datascienceclusters", "clusterqueue"}
        for call in command.call_args_list
        if call.args[0][1] == "get"
    )


def test_localqueue_setup_failure_does_not_abort_other_fixtures(prepared_cluster, tmp_path):
    args, values, _, _, command = prepared_cluster
    cluster_preflight(args, command, values)
    environment = RunEnvironment(args, "run-123", tmp_path, Mock())
    documents = []

    def create(filename, document):
        documents.append(document)
        if document["kind"] == "Namespace":
            document["metadata"]["annotations"] = {
                "openshift.io/sa.scc.supplemental-groups": "1001420000/10000"
            }
        if document["kind"] == "LocalQueue":
            raise RuntimeError("raw private response must not be exposed")
        document["metadata"]["uid"] = "uid-" + document["metadata"]["name"]
        return document

    environment.create = create
    environment.wait_for_fixtures = Mock()
    environment.provision()
    assert documents[-1]["kind"] == "Pod"
    assert workload(args.job_file, "with-kueue")["queue_name"] == "e2e"
    assert "LocalQueue" in environment.kueue_setup_error
    assert "private" not in environment.kueue_setup_error


@pytest.fixture
def queue_observer(monkeypatch, prepared_cluster):
    from tests.e2e import observer as module

    _, _, _, queue, _ = prepared_cluster
    result = object.__new__(Observer)
    result.project, result.queue = "new-test-project", None
    result.api, result.core, result.custom = Mock(), Mock(), Mock()
    result.inventory, result.listing = Mock(return_value=[]), Mock(return_value=[])
    result.preserved, result.uids, result.resources = [], set(), set()
    result.job_uid, result.saw_gpu, result.admitted = None, False, False
    result.core.read_namespace.return_value.metadata.labels = namespace_manifest(
        result.project, "run"
    )["metadata"]["labels"]
    result.custom.get_namespaced_custom_object.return_value = {"spec": {"clusterQueue": "default"}}
    result.custom.get_cluster_custom_object.return_value = queue
    discovery = Mock()
    discovery.get_api_versions.return_value.groups = [SimpleNamespace(name="kueue.x-k8s.io")]
    monkeypatch.setattr(module.client, "ApisApi", Mock(return_value=discovery))
    payload = json.loads((deploy.ROOT / "tests/e2e/job.example.json").read_text())
    payload["queue_name"] = "e2e"
    return result, payload, queue, discovery


def test_observer_accepts_operational_kueue_regardless_of_installation_method(queue_observer):
    observer, payload, _, _ = queue_observer
    observer.preflight(payload, "with-kueue")
    assert observer.queue == "e2e"
    observer.inventory.assert_called_once()


@pytest.mark.parametrize(
    "problem,expected",
    [
        ("absent", "Kueue API group is absent"),
        ("local-missing", "LocalQueue 'new-test-project/e2e': not found"),
        ("forbidden", "ClusterQueue 'default': observer permission denied"),
        ("inactive", "Active=False (FlavorNotFound)"),
        ("selector", "namespaceSelector mismatch"),
        ("resources", "does not cover requested resources: nvidia.com/gpu"),
        (
            "workloads",
            "Kueue Workloads in namespace 'new-test-project': observer permission denied",
        ),
    ],
)
def test_queue_failures_are_actionable_and_do_not_expose_raw_bodies(
    queue_observer, problem, expected
):
    observer, payload, queue, discovery = queue_observer
    if problem == "absent":
        discovery.get_api_versions.return_value.groups = []
    elif problem == "local-missing":
        observer.custom.get_namespaced_custom_object.side_effect = ApiException(
            status=404, reason="private-token"
        )
    elif problem == "forbidden":
        observer.custom.get_cluster_custom_object.side_effect = ApiException(
            status=403, reason="private-token"
        )
    elif problem == "inactive":
        queue["status"]["conditions"] = [
            {
                "type": "Active",
                "status": "False",
                "reason": "FlavorNotFound",
                "message": "private-token",
            }
        ]
    elif problem == "selector":
        queue["spec"]["namespaceSelector"]["matchLabels"]["other-policy"] = "required"
    elif problem == "resources":
        queue["spec"]["resourceGroups"][0]["coveredResources"].remove("nvidia.com/gpu")
    else:
        observer.listing.side_effect = ApiException(status=403, reason="private-token")
    with pytest.raises(ObserverError) as error:
        observer.preflight(payload, "with-kueue")
    assert expected in failure_reason(error.value)
    assert "private-token" not in failure_reason(error.value)
    observer.inventory.assert_not_called()


def test_pending_workload_conditions_are_available_for_timeout_diagnostics(queue_observer):
    observer, payload, _, _ = queue_observer
    observer.preflight(payload, "with-kueue")
    observer.custom.get_namespaced_custom_object.return_value = {
        "metadata": {"uid": "job-uid", "labels": {"kueue.x-k8s.io/queue-name": "e2e"}}
    }
    observer.listing.return_value = [
        {
            "metadata": {
                "name": "owned-workload",
                "uid": "workload-uid",
                "ownerReferences": [{"uid": "job-uid", "controller": True}],
            },
            "spec": {"queueName": "e2e"},
            "status": {
                "conditions": [
                    {
                        "type": "QuotaReserved",
                        "status": "False",
                        "reason": "Pending",
                        "message": "private-token",
                    },
                    {"type": "Admitted", "status": "False", "reason": "NoReservation"},
                ]
            },
        }
    ]
    observer.observe("test-job")
    assert "QuotaReserved=False (Pending)" in observer.queue_diagnostic()
    assert "Admitted=False (NoReservation)" in observer.queue_diagnostic()
    assert "private-token" not in observer.queue_diagnostic()


def test_failed_queue_fixture_continues_suite_and_preserves_failure_exit(jobfile, tmp_path, capsys):
    payload = json.loads(jobfile.read_text())
    payload["queue_name"] = "e2e"
    jobfile.write_text(json.dumps(payload))
    suite = tmp_path / "test_continuation.py"
    suite.write_text("""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tests.e2e import observer as module

@pytest.fixture
def api():
    return SimpleNamespace(project="unit-project")

@pytest.fixture(autouse=True)
def broken_queue(monkeypatch, request):
    result = SimpleNamespace(
        api=Mock(),
        preflight=Mock(side_effect=module.ObserverError(
            "LocalQueue 'unit-project/e2e': not found (HTTP 404)"
        )),
    )
    monkeypatch.setattr(module, "Observer", lambda project: result)
    yield
    if request.node.name == "test_a_queued":
        result.api.close.assert_called_once()
    else:
        result.api.close.assert_not_called()

def test_a_queued(observer):
    raise AssertionError("must fail before submission")

def test_b_independent():
    assert True
""")
    env = dict(
        {k: v for k, v in os.environ.items() if not k.startswith("TRAINING_SERVICE_")},
        TRAINING_SERVICE_URL="https://unit.invalid",
        TRAINING_SERVICE_PROJECT="unit-project",
        TRAINING_SERVICE_TOKEN="unit-placeholder",
        TRAINING_SERVICE_EVIDENCE_DIR=str(tmp_path / "results"),
        PYTEST_ADDOPTS="-x",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.e2e.conftest",
            str(suite),
            "--job-file",
            str(jobfile),
            "--queue-profile",
            "with-kueue",
            "--observe-cluster",
            "--maxfail=0",
            "--junitxml",
            str(tmp_path / "results/junit.xml"),
        ],
        cwd=deploy.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "1 passed, 1 error" in result.stdout
    report = json.loads((tmp_path / "results/tests.json").read_text())
    assert report["passed"] == report["failed"] == 1
    assert "LocalQueue 'unit-project/e2e': not found" in report["failures"][0]["reason"]
    assert (
        "LocalQueue 'unit-project/e2e': not found" in (tmp_path / "results/junit.xml").read_text()
    )
    terminal_summary({"outcome": "failed", "namespace": "unit-project"}, tmp_path / "results")
    output = capsys.readouterr().out
    assert "1 passed, 1 failed" in output
    assert "LocalQueue 'unit-project/e2e': not found" in output


@pytest.mark.parametrize(
    "problem",
    ["storage", "runtime", "private"],
)
def test_missing_deployment_prerequisites_stop_before_mutation(prepared_cluster, problem):
    args, values, _, _, command = prepared_cluster
    if problem == "storage":
        args.storage_class = None
    elif problem == "runtime":
        values["env"][0]["value"] = "YOUR_RUNTIME_IMAGE"
    else:
        values["imagePullSecrets"] = [{"name": "private-pull"}]
    with pytest.raises(ValueError):
        cluster_preflight(args, command, values)
    assert all(call.args[0][1] in {"get", "auth"} for call in command.call_args_list)


def test_cluster_preflight_rejects_existing_namespace(prepared_cluster):
    args, values, _, _, command = prepared_cluster
    query = command.side_effect
    command.side_effect = lambda cmd, **kw: (
        "namespace/existing" if cmd[:3] == ["oc", "get", "namespace"] else query(cmd, **kw)
    )
    with pytest.raises(ValueError, match="already exists"):
        cluster_preflight(args, command, values)


def test_environment_creates_queued_fixture_and_readonly_verifier(prepared_cluster, tmp_path):
    args, values, _, _, command = prepared_cluster
    cluster_preflight(args, command, values)
    environment = RunEnvironment(args, "run-123", tmp_path, Mock())
    documents = []

    def create(filename, document):
        documents.append(document)
        if document["kind"] == "Namespace":
            document["metadata"]["annotations"] = {
                "openshift.io/sa.scc.supplemental-groups": "1001420000/10000"
            }
        document["metadata"]["uid"] = "uid-" + document["metadata"]["name"]
        return document

    environment.create = create
    environment.wait_for_fixtures = Mock()
    environment.provision()
    assert [d["kind"] for d in documents] == [
        "Namespace",
        "LocalQueue",
        "PersistentVolumeClaim",
        "ConfigMap",
        "Pod",
    ]
    assert documents[1]["spec"]["clusterQueue"] == "default"
    assert documents[2]["spec"]["accessModes"] == ["ReadWriteMany"]
    pod = documents[-1]["spec"]
    assert not pod["automountServiceAccountToken"]
    assert pod["securityContext"]["fsGroup"] == 1001420000
    assert pod["securityContext"]["fsGroupChangePolicy"] == "OnRootMismatch"
    assert pod["containers"][0]["volumeMounts"][0]["readOnly"]
    assert "kueue.x-k8s.io/queue-name" not in documents[-1]["metadata"]["labels"]
    payload = workload(args.job_file, "with-kueue")
    assert payload["queue_name"] == "e2e"
    assert payload["model"]["uri"].startswith("pvc://e2e-storage/")
    assert args.artifact_pod == "e2e-output-verifier"


@pytest.mark.parametrize("annotation", ["supplemental-groups", "uid-range"])
def test_fixture_uses_namespace_allocated_group(prepared_cluster, tmp_path, annotation):
    environment = RunEnvironment(prepared_cluster[0], "run", tmp_path, Mock())
    assert (
        environment.fixture_group(
            {
                "metadata": {
                    "annotations": {
                        "openshift.io/sa.scc." + annotation: "1001420000/10000,1001430000/10000"
                    }
                }
            }
        )
        == 1001420000
    )
    environment.command.assert_not_called()


def test_fixture_waits_for_namespace_security_allocation(monkeypatch, prepared_cluster, tmp_path):
    from tests.e2e import environment as module

    command = Mock(
        return_value=json.dumps(
            {
                "metadata": {
                    "annotations": {"openshift.io/sa.scc.supplemental-groups": "1001420000/10000"}
                }
            }
        )
    )
    environment = RunEnvironment(prepared_cluster[0], "run", tmp_path, command)
    monkeypatch.setattr(module.time, "sleep", Mock())
    assert environment.fixture_group({"metadata": {}}) == 1001420000
    command.assert_called_once()


@pytest.mark.parametrize("allocated", ["0/10000", "1001420000/0", "not-a-range"])
def test_fixture_rejects_invalid_or_root_group(prepared_cluster, tmp_path, allocated):
    environment = RunEnvironment(prepared_cluster[0], "run", tmp_path, Mock())
    with pytest.raises(RuntimeError, match="invalid allocated filesystem group"):
        environment.fixture_group(
            {"metadata": {"annotations": {"openshift.io/sa.scc.supplemental-groups": allocated}}}
        )


@pytest.mark.parametrize("container_kind", ["initContainerStatuses", "containerStatuses"])
def test_fixture_failure_exits_immediately_with_log_instructions(
    monkeypatch, prepared_cluster, tmp_path, container_kind
):
    from tests.e2e import environment as module

    pod = {
        "metadata": {"uid": "fixture-uid"},
        "status": {
            "phase": "Failed",
            container_kind: [
                {
                    "name": "prepare",
                    "state": {"terminated": {"exitCode": 1, "message": "private response"}},
                }
            ],
        },
    }
    environment = RunEnvironment(
        prepared_cluster[0], "run", tmp_path, Mock(return_value=json.dumps(pod))
    )
    environment.pod_uid = "fixture-uid"
    sleep = Mock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="exited with code 1") as error:
        environment.wait_for_fixtures()
    assert "oc logs -n new-test-project e2e-output-verifier -c prepare" in str(error.value)
    assert "private" not in str(error.value)
    sleep.assert_not_called()


def test_fixture_pending_progress_then_ready(monkeypatch, prepared_cluster, tmp_path, capsys):
    from tests.e2e import environment as module

    pending = {"metadata": {"uid": "fixture-uid"}, "status": {"phase": "Pending"}}
    ready = {
        "metadata": {"uid": "fixture-uid"},
        "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}]},
    }
    environment = RunEnvironment(
        prepared_cluster[0],
        "run",
        tmp_path,
        Mock(side_effect=[json.dumps(pending), json.dumps(ready)]),
    )
    environment.pod_uid = "fixture-uid"
    monkeypatch.setattr(module.time, "sleep", Mock())
    environment.wait_for_fixtures()
    assert "Waiting for fixtures: Pending" in capsys.readouterr().out


def test_fixture_readiness_deadline_is_bounded(monkeypatch, prepared_cluster, tmp_path):
    from tests.e2e import environment as module

    args = prepared_cluster[0]
    args.fixture_timeout = 1
    pending = {"metadata": {"uid": "fixture-uid"}, "status": {"phase": "Pending"}}
    environment = RunEnvironment(args, "run", tmp_path, Mock(return_value=json.dumps(pending)))
    environment.pod_uid = "fixture-uid"
    monkeypatch.setattr(module.time, "monotonic", Mock(side_effect=[0, 1]))
    with pytest.raises(RuntimeError, match="Fixture readiness deadline exceeded"):
        environment.wait_for_fixtures()


@pytest.mark.parametrize(
    "image", ["quay.io/other/service@sha256:" + "a" * 64, "quay.io/test/training-service:latest"]
)
def test_reused_image_must_be_digest_pinned_in_selected_repository(prepared_cluster, image):
    args = prepared_cluster[0]
    args.reuse_image = image
    with pytest.raises(ValueError, match="must pin a sha256 digest"):
        deploy.preflight(args)


def test_reuse_preflight_needs_no_podman_or_push_credentials(monkeypatch, prepared_cluster):
    args, _, _, _, _ = prepared_cluster
    args.reuse_image = args.image_repository + "@sha256:" + "a" * 64
    args.values.write_text("env: []\n")
    monkeypatch.setenv("TRAINING_SERVICE_TOKEN", "unit-placeholder")
    monkeypatch.setattr(deploy, "cluster_preflight", Mock())
    which = Mock(side_effect=lambda name: None if name == "podman" else "/usr/bin/" + name)
    monkeypatch.setattr(deploy.shutil, "which", which)

    def query(cmd, **kwargs):
        assert cmd[0] != "podman"
        if cmd[:3] == ["oc", "get", "nodes"]:
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"labels": {"kubernetes.io/arch": "amd64"}},
                            "status": {"allocatable": {"nvidia.com/gpu": "1"}},
                        }
                    ]
                }
            )
        return "yes"

    monkeypatch.setattr(deploy, "command", query)
    deploy.preflight(args)
    assert [call.args[0] for call in which.call_args_list] == ["helm", "oc", "git"]


def test_fixture_write_failure_happens_before_model_download(monkeypatch):
    import runpy
    import tempfile

    download = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(os, "umask", Mock())
    monkeypatch.setattr(
        tempfile, "TemporaryFile", Mock(side_effect=PermissionError("private path"))
    )
    with pytest.raises(RuntimeError, match="PVC is not writable") as error:
        runpy.run_path(str(deploy.ROOT / "tests/e2e/fixtures/prepare.py"))
    assert "private" not in str(error.value)
    download.assert_not_called()


def test_fixture_cleanup_refuses_replaced_namespace(prepared_cluster, tmp_path):
    args = prepared_cluster[0]
    command = Mock(
        return_value=json.dumps(
            {"metadata": {"uid": "replacement", "labels": {RUN_LABEL: "run-123"}}}
        )
    )
    environment = RunEnvironment(args, "run-123", tmp_path, command)
    environment.namespace_uid = "original"
    with pytest.raises(RuntimeError, match="ownership changed"):
        environment.cleanup_setup_pod()
    assert command.call_count == 1


@pytest.mark.parametrize("replacement", [False, True])
def test_fixture_cleanup_uses_pod_uid_precondition(
    monkeypatch, prepared_cluster, tmp_path, replacement
):
    from kubernetes import client, config

    args = prepared_cluster[0]
    command = Mock(
        side_effect=[
            json.dumps({"metadata": {"uid": "namespace-uid", "labels": {RUN_LABEL: "run"}}}),
            json.dumps(
                {
                    "metadata": {
                        "uid": "replacement" if replacement else "pod-uid",
                        "labels": {RUN_LABEL: "run"},
                    }
                }
            ),
            "",
        ]
    )
    environment = RunEnvironment(args, "run", tmp_path, command)
    environment.namespace_uid, environment.pod_uid = "namespace-uid", "pod-uid"
    core = Mock()
    monkeypatch.setattr(config, "new_client_from_config", MagicMock())
    monkeypatch.setattr(client, "CoreV1Api", Mock(return_value=core))
    if replacement:
        with pytest.raises(RuntimeError, match="ownership changed"):
            environment.cleanup_setup_pod()
        core.delete_namespaced_pod.assert_not_called()
    else:
        environment.cleanup_setup_pod()
        assert core.delete_namespaced_pod.call_args.kwargs["body"].preconditions.uid == "pod-uid"
        assert command.call_args.args[0][1] == "wait"


def test_preflight_only_never_builds_pushes_or_provisions(monkeypatch, tmp_path, capsys):
    args = deploy.arguments(
        [
            "--image-repository",
            "quay.io/test/training-service",
            "--values",
            "unused.yaml",
            "--results-dir",
            str(tmp_path),
            "--preflight-only",
        ]
    )
    preflight, command, environment = Mock(), Mock(), Mock()
    monkeypatch.setattr(deploy, "preflight", preflight)
    tls = Mock(return_value={})
    monkeypatch.setattr(deploy, "prepare_tls", tls)
    monkeypatch.setattr(deploy, "command", command)
    monkeypatch.setattr(deploy, "RunEnvironment", environment)
    environment.return_value.namespace_uid = None
    environment.return_value.kueue_setup_error = None
    deploy.run(args)
    preflight.assert_called_once()
    tls.assert_called_once()
    assert args.namespace.startswith("training-e2e-")
    command.assert_not_called()
    environment.return_value.provision.assert_not_called()
    assert "PREFLIGHT_PASSED" in capsys.readouterr().out


def test_preflight_failure_prints_summary_without_creating_resources(monkeypatch, tmp_path, capsys):
    args = deploy.arguments(
        [
            "--image-repository",
            "quay.io/test/training-service",
            "--values",
            "unused.yaml",
            "--results-dir",
            str(tmp_path),
        ]
    )
    monkeypatch.setattr(deploy, "preflight", Mock(side_effect=ValueError("StorageClass missing")))
    command = Mock()
    monkeypatch.setattr(deploy, "command", command)
    with pytest.raises(ValueError, match="StorageClass missing"):
        deploy.run(args)
    command.assert_not_called()
    assert "Result: FAILED" in capsys.readouterr().out


def test_invalid_runner_inputs_fail_before_commands(monkeypatch, tmp_path):
    args = deploy.arguments(
        [
            "--image-repository",
            "not-quay/image",
            "--namespace",
            "test-project",
            "--values",
            str(tmp_path / "values.yaml"),
        ]
    )
    command = Mock()
    monkeypatch.setattr(deploy, "command", command)
    with pytest.raises(ValueError, match="quay.io"):
        deploy.preflight(args)
    command.assert_not_called()


def test_chart_mounts_ca_only_when_configured():
    # Helm rendering, no cluster/credentials needed; current CI already installs Helm.
    import shutil

    if not shutil.which("helm"):
        pytest.skip("helm is not installed")
    plain = deploy.command(["helm", "template", "e2e", deploy.CHART])
    configured = deploy.command(
        [
            "helm",
            "template",
            "e2e",
            deploy.CHART,
            "-f",
            deploy.ROOT / "tests/e2e/values.example.yaml",
        ]
    )
    assert "volumeMounts:" not in plain and "volumes:" not in plain
    assert "volumeMounts:" in configured and "readOnly: true" in configured


@pytest.mark.parametrize("kind", ["foreign", "active"])
def test_cleanup_never_deletes_foreign_or_active_workloads(kind):
    calls = []
    payload = {"name": "e2e-owned"}

    def transport(request):
        calls.append(request.method)
        return httpx.Response(
            200,
            json={
                "status": "RUNNING" if kind == "active" else "SUCCEEDED",
                "submission": payload if kind == "active" else {"name": "foreign"},
            },
        )

    with httpx.Client(
        base_url="https://unit.invalid", transport=httpx.MockTransport(transport)
    ) as http:
        api = LiveAPI(http, "test-project")
        with pytest.raises(pytest.fail.Exception):
            api.cleanup("e2e-owned", 0, payload)
    assert calls == ["GET"]


@pytest.mark.parametrize(
    "body,passes",
    [
        (
            'event: log\ndata: {"message":"first"}\n\nevent: log\ndata: {"message":"second"}\n\n',
            True,
        ),
        ('event: log\ndata: {"message":"same"}\n\nevent: log\ndata: {"message":"same"}\n\n', False),
        ('event: error\ndata: {"code":"logs_unavailable","message":"failed"}\n\n', False),
    ],
)
def test_stream_checks_distinct_sse_events_and_rejects_error_events(body, passes):
    def transport(request):
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)

    with httpx.Client(
        base_url="https://unit.invalid",
        headers={"Authorization": "Bearer unit-token"},
        transport=httpx.MockTransport(transport),
    ) as http:
        api = LiveAPI(http, "test-project")
        if passes:
            assert follow_logs(api, "/jobs/e2e-owned", 1) == 2
        else:
            with pytest.raises((pytest.fail.Exception, AssertionError)):
                follow_logs(api, "/jobs/e2e-owned", 1)


def test_runner_missing_http_credential_fails_before_build(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAINING_SERVICE_TOKEN", raising=False)
    args = deploy.arguments(
        [
            "--image-repository",
            "quay.io/test/service",
            "--namespace",
            "test-project",
            "--values",
            str(tmp_path / "values.yaml"),
        ]
    )
    run = Mock()
    monkeypatch.setattr(deploy, "command", run)
    with pytest.raises(ValueError, match="TRAINING_SERVICE_TOKEN"):
        deploy.preflight(args)
    run.assert_not_called()


def test_all_skipped_live_selection_is_not_a_passing_gate():
    # No HTTP request: these three prepared-fault cases skip before making a call.
    env = dict(
        {k: v for k, v in os.environ.items() if not k.startswith("TRAINING_SERVICE_")},
        TRAINING_SERVICE_URL="https://unit.invalid",
        TRAINING_SERVICE_PROJECT="unit-project",
        TRAINING_SERVICE_TOKEN="unit-placeholder",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/e2e",
            "-k",
            "invalid_kubernetes_url",
        ],
        cwd=deploy.ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "3 skipped" in result.stdout
