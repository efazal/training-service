"""Credential-free runner/observer safety tests, not substitutes for live E2E."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import httpx
import pytest
import yaml
from kubernetes.client.exceptions import ApiException
from tests.e2e import conftest as live_config
from tests.e2e import deploy, support
from tests.e2e import test_training_jobs as training_jobs
from tests.e2e.algorithms import ALGORITHMS, algorithm_workload
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
    payload = json.loads((deploy.ROOT / "tests/e2e/examples/job.example.json").read_text())
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


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithm_matrix_workloads_match_submission_validation(
    jobfile, algorithm, runtime_catalog
):
    from training_service import catalog
    from training_service.errors import ApiError
    from training_service.jobs import validate_algorithm_parameters
    from training_service.ray_entrypoint import get_backend

    original = json.loads(jobfile.read_text())
    original["dataset"]["uri"] = "pvc://e2e-storage/datasets/tiny-chat.jsonl"
    payload = algorithm_workload(original, algorithm)
    assert original["algorithm"]["name"] == "lora_sft"
    assert payload["algorithm"]["name"] == algorithm
    assert payload["dataset"]["uri"].startswith("pvc://e2e-storage/datasets/")
    if algorithm == "lora_grpo":
        # The math profile needs a runtime that explicitly advertises reward_type.
        with pytest.raises(ApiError, match="reward_type"):
            validate_algorithm_parameters(algorithm, payload["algorithm"]["parameters"])
        metadata = json.loads(runtime_catalog.read_text())
        descriptor = next(a for a in metadata["algorithms"] if a["name"] == algorithm)
        descriptor["parameter_schema"]["properties"]["reward_type"] = {"enum": ["math"]}
        runtime_catalog.write_text(json.dumps(metadata))
        catalog._load_catalog.cache_clear()
    parameters = validate_algorithm_parameters(algorithm, payload["algorithm"]["parameters"])
    resources = payload["resources"]
    get_backend(algorithm).validate_resources(
        resources["gpus_per_node"], resources["num_nodes"], resources["gpu_type"], parameters
    )


def test_algorithm_matrix_rejects_unshared_grpo_output(jobfile):
    payload = json.loads(jobfile.read_text())
    payload["dataset"]["uri"] = "pvc://e2e-storage/datasets/tiny-chat.jsonl"
    payload["output"]["destination_uri"] = "s3://credentials/bucket/output/"
    with pytest.raises(ValueError, match="shared PVC output"):
        algorithm_workload(payload, "lora_grpo")


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


@pytest.mark.parametrize("answer,exit_code,expected", [("yes\n", 0, "yes\n"), ("no\n", 1, "no")])
def test_can_i_denial_is_a_normal_permission_answer(monkeypatch, answer, exit_code, expected):
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [], exit_code, stdout=answer, stderr="private-token"
            )
        ),
    )
    assert deploy.command(["oc", "auth", "can-i", "create", "namespaces"]) == expected


def test_can_i_connection_failure_is_not_treated_as_denied_permission(monkeypatch):
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="private-token")),
    )
    with pytest.raises(RuntimeError, match="Raw output withheld"):
        deploy.command(["oc", "auth", "can-i", "create", "namespaces"])


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("teardown_problem", [None, "timeout", "interrupt"])
def test_pytest_deadline_allows_fixture_teardown_before_killing(
    monkeypatch, interrupted, teardown_problem
):
    error = KeyboardInterrupt() if interrupted else subprocess.TimeoutExpired("pytest", 1)
    process = MagicMock()
    process.__enter__.return_value = process
    process.communicate.side_effect = [
        error,
        *(
            [subprocess.TimeoutExpired("teardown", 7)]
            if teardown_problem == "timeout"
            else [KeyboardInterrupt()]
            if teardown_problem == "interrupt"
            else []
        ),
        ("", ""),
    ]
    popen = Mock(return_value=process)
    monkeypatch.setattr(deploy.subprocess, "Popen", popen)
    with pytest.raises(KeyboardInterrupt if interrupted else RuntimeError):
        deploy.command(
            [sys.executable, "-m", "pytest", "tests/e2e"],
            timeout=1,
            cleanup_timeout=7,
            capture=False,
        )
    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert process.communicate.call_args_list[1].kwargs["timeout"] == 7
    assert process.kill.called == (teardown_problem is not None)
    assert popen.call_args.kwargs["start_new_session"] is True


def test_sigterm_enters_runner_cleanup_instead_of_abandoning_it():
    script = """
import os, signal
from tests.e2e import deploy

deploy.arguments = lambda: None
def run(args):
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    finally:
        print("owned-cleanup-ran", flush=True)
deploy.run = run
previous = signal.getsignal(signal.SIGTERM)
result = deploy.main()
assert signal.getsignal(signal.SIGTERM) == previous
raise SystemExit(result)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=deploy.ROOT, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 130
    assert "owned-cleanup-ran" in result.stdout


def test_real_pytest_timeout_runs_fixture_teardown(tmp_path):
    ready, finished = tmp_path / "ready", tmp_path / "finished"
    suite = tmp_path / "test_interrupt.py"
    suite.write_text(f"""
from pathlib import Path
import time
import pytest

@pytest.fixture
def resource():
    Path({str(ready)!r}).touch()
    try:
        yield
    finally:
        Path({str(finished)!r}).touch()

def test_wait(resource):
    time.sleep(60)
""")
    with pytest.raises(RuntimeError, match="could not finish"):
        deploy.command(
            [sys.executable, "-m", "pytest", "-c", "/dev/null", str(suite)],
            timeout=5,
            cleanup_timeout=5,
        )
    assert ready.is_file(), "Fixture must have started before the test deadline"
    assert finished.is_file(), "Handled interrupt must execute fixture teardown"


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_setup_failure_or_interruption_still_cleans_owned_namespace(
    monkeypatch, tmp_path, interrupted
):
    args = deploy.arguments(
        [
            "--image-repository",
            "quay.io/test/training-service",
            "--reuse-image",
            "quay.io/test/training-service@sha256:" + "a" * 64,
            "--values",
            "unused.yaml",
            "--results-dir",
            str(tmp_path),
        ]
    )
    monkeypatch.setattr(deploy, "preflight", Mock())
    monkeypatch.setattr(deploy, "prepare_tls", Mock(return_value={}))
    monkeypatch.setattr(deploy, "command", Mock(return_value="commit"))
    environment = Mock(namespace_uid="owned-uid", kueue_setup_error=None)
    environment.provision.side_effect = (
        KeyboardInterrupt() if interrupted else ValueError("setup failed")
    )
    environment.cleanup.side_effect = lambda release, deployed, evidence: evidence.update(
        namespace_cleanup="removed"
    )
    monkeypatch.setattr(deploy, "RunEnvironment", Mock(return_value=environment))
    with pytest.raises(KeyboardInterrupt if interrupted else ValueError):
        deploy.run(args)
    assert environment.cleanup.call_args.args[1] is False
    report = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert report["outcome"] == "failed"
    assert report["namespace_cleanup"] == "removed"


@pytest.mark.parametrize(
    "failure,keep,missing_evidence",
    [(False, False, False), (False, True, False), (True, False, False), (False, False, True)],
)
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("cleanup_failure", [None, "error", "interrupt"])
@pytest.mark.parametrize("matrix", [False, True])
def test_runner_build_push_digest_install_test_cleanup(
    monkeypatch, tmp_path, failure, keep, missing_evidence, reuse, private, cleanup_failure, matrix
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
            *(["--algorithm-matrix"] if matrix else []),
            *(["--keep-resources"] if keep else []),
            *(["--use-podman-auth"] if private else []),
            *(
                ["--reuse-image", "quay.io/test/training-service@sha256:" + "a" * 64]
                if reuse
                else []
            ),
        ]
    )
    monkeypatch.setattr(deploy, "preflight", Mock())
    args.runtime_image = "quay.io/test/ray:tested"
    args.ingress_ca_bundle = tmp_path / "generated-ingress-ca.pem"
    monkeypatch.setattr(deploy, "prepare_tls", Mock(return_value={"ingress_ca_source": "test"}))
    monkeypatch.setattr(deploy, "wait_ready", Mock())
    environment = Mock(namespace_uid="test-namespace-uid", kueue_setup_error=None)
    environment.pull_secret.name = "unit-pull-secret"
    environment.pull_secret.cleanup.return_value = "removed"

    def cleanup(release, deployed, evidence):
        if cleanup_failure:
            if cleanup_failure == "interrupt":
                raise KeyboardInterrupt
            raise RuntimeError("namespace deletion failed")
        evidence.update(namespace_cleanup="removed", deployment_cleanup="removed")

    environment.cleanup.side_effect = cleanup

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
            return json.dumps(
                {"items": [{"spec": {"imagePullSecrets": [{"name": "unit-pull-secret"}]}}]}
            )
        if command[:2] == ["oc", "exec"]:
            return json.dumps(
                {"training-service": "0.1.0", "ray": "2.55.1", "codeflare-sdk": "0.38.2"}
            )
        if command[:2] == ["oc", "version"]:
            return "{}"
        if "pytest" in command:
            assert kwargs["cleanup_timeout"] == args.cleanup_timeout
            assert kwargs["env"]["TRAINING_SERVICE_CA_BUNDLE"] == str(args.ingress_ca_bundle)
            assert ("--algorithm-matrix" in command) == matrix
            if failure:
                raise RuntimeError("live tests failed")
            if missing_evidence:
                return ""
            results = Path(kwargs["env"]["TRAINING_SERVICE_EVIDENCE_DIR"])
            names = ALGORITHMS if matrix else (None,)
            for name in names:
                filename = f"lifecycle-{name}.json" if name else "lifecycle.json"
                (results / filename).write_text(
                    json.dumps(
                        {
                            "succeeded": True,
                            "queue_admitted": True,
                            "gpu_cleanup": True,
                            "snapshot_logs": True,
                            "sse": not matrix,
                            "artifacts": {"nonempty_output_files": 1},
                        }
                    )
                )
        return ""

    monkeypatch.setattr(deploy, "command", run)
    failed = failure or missing_evidence
    cleanup_failed = bool(cleanup_failure and not keep)
    if failed or cleanup_failed:
        with pytest.raises(
            RuntimeError, match="live tests failed|evidence is missing|cleanup incomplete"
        ):
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
    assert ("imagePullSecrets[0].name=unit-pull-secret" in install) == private
    invocation = next(c for c in commands if "pytest" in c)
    assert invocation[invocation.index("--queue-profile") + 1] == "with-kueue"
    assert "--maxfail=0" in invocation
    assert {"--job-file", "--artifact-pod", "--observe-cluster"} <= set(invocation)
    assert ("--check-stream" in invocation) != matrix
    environment.provision.assert_called_once()
    assert environment.cleanup.called == (not keep)
    report = json.loads(next((tmp_path / "results").glob("*/run.json")).read_text())
    assert report["algorithm_matrix"] is matrix
    assert report["outcome"] == (
        "cleanup_failed" if cleanup_failed else "failed" if failed else "passed"
    )
    assert report["image"].endswith("@" + digest)
    assert report["runtime_image"] == args.runtime_image
    assert report["image_reused"] == reuse
    assert environment.pull_secret.cleanup.called == (private and (keep or cleanup_failed))
    if private:
        assert report["registry_secret_cleanup"] == (
            "removed" if keep or cleanup_failed else "removed with namespace"
        )
    assert report["namespace_cleanup"] == (
        "retained (--keep-resources)"
        if keep
        else "incomplete; inspect test namespace"
        if cleanup_failed
        else "removed"
    )


@pytest.fixture
def prepared_cluster(tmp_path):
    values = {
        "trainingRuntime": {"image": "quay.io/test/ray:tested"},
        "env": [
            {"name": "TRAINING_SERVICE_RAY_GATEWAY_URL", "value": "https://ray.example.com"},
        ],
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


def test_cluster_preflight_uses_chart_runtime_default(prepared_cluster):
    args, values, _, _, command = prepared_cluster
    values.pop("trainingRuntime")
    cluster_preflight(args, command, values)
    defaults = yaml.safe_load((deploy.CHART / "values.yaml").read_text())
    assert args.runtime_image == defaults["trainingRuntime"]["image"]


def test_cluster_preflight_accepts_legacy_runtime_environment_value(prepared_cluster):
    args, values, _, _, command = prepared_cluster
    runtime = values.pop("trainingRuntime")["image"]
    values["env"].append({"name": "TRAINING_RAY_IMAGE", "value": runtime})
    cluster_preflight(args, command, values)
    assert args.runtime_image == runtime


def test_cluster_preflight_rejects_duplicate_runtime_configuration(prepared_cluster):
    args, values, _, _, command = prepared_cluster
    values["env"].append({"name": "TRAINING_RAY_IMAGE", "value": "other-image"})
    with pytest.raises(ValueError, match="trainingRuntime.image.*env"):
        cluster_preflight(args, command, values)


@pytest.mark.parametrize("image", ["", None, 123, "YOUR_RUNTIME_IMAGE"])
def test_cluster_preflight_requires_a_concrete_runtime_image(prepared_cluster, image):
    args, values, _, _, command = prepared_cluster
    values["trainingRuntime"]["image"] = image
    with pytest.raises(ValueError, match="trainingRuntime.image"):
        cluster_preflight(args, command, values)


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


def test_cleanup_permission_failure_names_required_resource(prepared_cluster):
    args, values, _, _, command = prepared_cluster
    original = command.side_effect

    def denied(cmd, **kwargs):
        if cmd[:5] == ["oc", "auth", "can-i", "delete", "namespaces"]:
            return "no"
        return original(cmd, **kwargs)

    command.side_effect = denied
    with pytest.raises(ValueError, match="delete access to namespaces for cleanup"):
        cluster_preflight(args, command, values)


@pytest.mark.parametrize("option", ["--keep-resources", "--keep-deployment"])
def test_debug_retention_option_and_alias(prepared_cluster, option):
    args = deploy.arguments(
        [
            "--image-repository",
            "quay.io/test/training-service",
            "--values",
            "unused.yaml",
            option,
        ]
    )
    assert args.keep_resources is True


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
    payload = json.loads((deploy.ROOT / "tests/e2e/examples/job.example.json").read_text())
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
        values["trainingRuntime"]["image"] = "YOUR_RUNTIME_IMAGE"
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


@pytest.mark.parametrize("private", [False, True])
def test_environment_creates_queued_fixture_and_readonly_verifier(
    prepared_cluster, tmp_path, monkeypatch, private
):
    from tests.e2e import environment as module

    args, values, _, _, command = prepared_cluster
    args.use_podman_auth = private
    secret = Mock(name="pull-secret")
    secret.name = "unit-pull-secret"
    factory = Mock(return_value=secret)
    monkeypatch.setattr(module, "PullSecret", factory)
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
    assert pod.get("imagePullSecrets", []) == ([{"name": "unit-pull-secret"}] if private else [])
    assert factory.called == private
    assert secret.create.called == private
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
    environment = RunEnvironment(
        prepared_cluster[0], "run", tmp_path, Mock(return_value='{"items":[]}')
    )
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


@pytest.mark.parametrize("token_source", ["override", "oc"])
def test_reuse_preflight_needs_no_podman_or_push_credentials(
    monkeypatch, prepared_cluster, token_source
):
    args, _, _, _, _ = prepared_cluster
    args.reuse_image = args.image_repository + "@sha256:" + "a" * 64
    args.values.write_text("env: []\n")
    if token_source == "override":
        monkeypatch.setenv("TRAINING_SERVICE_TOKEN", "unit-placeholder")
    else:
        monkeypatch.delenv("TRAINING_SERVICE_TOKEN", raising=False)
    token_lookup = Mock(return_value=subprocess.CompletedProcess([], 0, "unit-placeholder\n", ""))
    monkeypatch.setattr(support.subprocess, "run", token_lookup)
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
    assert os.environ["TRAINING_SERVICE_TOKEN"] == "unit-placeholder"
    assert token_lookup.call_count == (token_source == "oc")


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


@pytest.fixture
def cleanup_environment(monkeypatch, prepared_cluster, tmp_path):
    from tests.e2e import environment as module

    environment = RunEnvironment(
        prepared_cluster[0], "run", tmp_path, Mock(return_value='{"items":[]}')
    )
    environment.namespace_uid = "namespace-uid"
    environment.namespace_creation_attempted = True
    core = Mock()
    namespace = SimpleNamespace(
        metadata=SimpleNamespace(uid="namespace-uid", labels={RUN_LABEL: "run"})
    )
    core.read_namespace.side_effect = [namespace, ApiException(status=404)]
    monkeypatch.setattr(module, "kubeconfig_client", MagicMock())
    monkeypatch.setattr(module.client, "CoreV1Api", Mock(return_value=core))
    return environment, core, namespace


@pytest.mark.parametrize("replacement", ["uid", "label"])
def test_cleanup_refuses_replaced_or_foreign_namespace(cleanup_environment, replacement):
    environment, core, current = cleanup_environment
    if replacement == "uid":
        current.metadata.uid = "replacement"
    else:
        current.metadata.labels[RUN_LABEL] = "another-run"
    with pytest.raises(RuntimeError, match="ownership changed"):
        environment.cleanup("unique-release", True, {})
    core.delete_namespace.assert_not_called()
    environment.command.assert_not_called()


@pytest.mark.parametrize("deployed", [False, True])
@pytest.mark.parametrize("ambiguous_post", [False, True])
def test_cleanup_uninstalls_helm_before_uid_guarded_namespace_delete(
    cleanup_environment, deployed, ambiguous_post
):
    environment, core, _ = cleanup_environment
    if ambiguous_post:
        environment.namespace_uid = None
    operations = Mock()
    operations.attach_mock(environment.command, "helm")
    operations.attach_mock(core.delete_namespace, "delete")
    evidence = {}
    environment.cleanup("unique-release", deployed, evidence)
    assert evidence["namespace_cleanup"] == "removed"
    assert "test PVC" in evidence["fixture_cleanup"]
    assert environment.namespace_uid == "namespace-uid"
    assert core.delete_namespace.call_args.kwargs["body"].preconditions.uid == "namespace-uid"
    assert environment.command.called == deployed
    if deployed:
        assert operations.mock_calls[0].args[0][:3] == ["helm", "uninstall", "unique-release"]
        assert evidence["deployment_cleanup"] == "removed"


def test_cleanup_does_not_delete_namespace_if_helm_uninstall_fails(cleanup_environment):
    environment, core, _ = cleanup_environment
    environment.command.side_effect = RuntimeError("helm uninstall failed")
    with pytest.raises(RuntimeError, match="helm uninstall failed"):
        environment.cleanup("unique-release", True, {})
    core.delete_namespace.assert_not_called()


def test_cleanup_does_not_hide_leftover_cluster_scoped_rbac(cleanup_environment):
    environment, core, _ = cleanup_environment
    environment.command.return_value = '{"items":[{"kind":"ClusterRole"}]}'
    with pytest.raises(RuntimeError, match="RBAC remains"):
        environment.cleanup("unique-release", True, {})
    core.delete_namespace.assert_not_called()


def test_cleanup_never_claims_success_for_pending_deletion(cleanup_environment, monkeypatch):
    from tests.e2e import environment as module

    environment, core, current = cleanup_environment
    core.read_namespace.side_effect = None
    core.read_namespace.return_value = current
    monkeypatch.setattr(module.time, "monotonic", Mock(side_effect=[0, 999]))
    with pytest.raises(RuntimeError, match="deletion deadline"):
        environment.cleanup("unique-release", False, {})


def test_cleanup_is_noop_before_namespace_creation(cleanup_environment):
    environment, core, _ = cleanup_environment
    environment.namespace_uid = None
    environment.namespace_creation_attempted = False
    evidence = {}
    environment.cleanup("unique-release", False, evidence)
    core.read_namespace.assert_not_called()
    core.delete_namespace.assert_not_called()
    assert evidence["namespace_cleanup"] == "not created"


@pytest.mark.parametrize("deployed", [False, True])
def test_cleanup_missing_namespace_is_not_proof_of_helm_cleanup(cleanup_environment, deployed):
    environment, core, _ = cleanup_environment
    core.read_namespace.side_effect = ApiException(status=404)
    evidence = {}
    if deployed:
        with pytest.raises(RuntimeError, match="cluster-scoped"):
            environment.cleanup("unique-release", deployed, evidence)
    else:
        environment.cleanup("unique-release", deployed, evidence)
        assert evidence["namespace_cleanup"] == "already absent"
    core.delete_namespace.assert_not_called()


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


@pytest.mark.parametrize("route_enabled", [True, False])
def test_chart_organization_image_and_route_defaults(route_enabled):
    import shutil

    if not shutil.which("helm"):
        pytest.skip("helm is not installed")
    command = ["helm", "template", "e2e", deploy.CHART]
    if not route_enabled:
        command.extend(["--set", "route.enabled=false"])
    documents = list(yaml.safe_load_all(deploy.command(command)))
    deployment = next(d for d in documents if d["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    api = pod["containers"][0]
    assert api["image"] == "quay.io/modh/training-service:0.1.0"
    runtime = "quay.io/modh/ray:2.58.0-py312-cu130-th0112"
    assert pod["initContainers"][0]["image"] == runtime
    assert [e for e in api["env"] if e["name"] == "TRAINING_RAY_IMAGE"] == [
        {"name": "TRAINING_RAY_IMAGE", "value": runtime}
    ]
    assert not pod.get("imagePullSecrets")
    routes = [d for d in documents if d["kind"] == "Route"]
    assert len(routes) == int(route_enabled)
    if route_enabled:
        assert "host" not in routes[0]["spec"]
        assert routes[0]["spec"]["tls"] == {
            "termination": "edge",
            "insecureEdgeTerminationPolicy": "Redirect",
        }


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
            deploy.ROOT / "tests/e2e/examples/values.example.yaml",
        ]
    )
    assert "name: training-service-tmp" in plain
    assert "mountPath: /tmp" in plain
    assert "value: /tmp/huggingface" in plain
    assert "name: ray-gateway-ca" not in plain
    assert "name: ray-gateway-ca" in configured and "readOnly: true" in configured
    plain_pod = next(d for d in yaml.safe_load_all(plain) if d["kind"] == "Deployment")["spec"][
        "template"
    ]["spec"]
    configured_pod = next(d for d in yaml.safe_load_all(configured) if d["kind"] == "Deployment")[
        "spec"
    ]["template"]["spec"]
    assert "ray-gateway-ca" not in {v["name"] for v in plain_pod["volumes"]}
    assert "ray-gateway-ca" in {v["name"] for v in configured_pod["volumes"]}


@pytest.mark.parametrize("legacy", [False, True])
def test_chart_wires_training_runtime_image_once(tmp_path, legacy):
    import shutil

    if not shutil.which("helm"):
        pytest.skip("helm is not installed")
    image = "quay.io/test/ray@sha256:" + "a" * 64
    values = {
        "env": [{"name": "UNRELATED_SETTING", "value": "preserved"}],
        "imagePullSecrets": [{"name": "private-runtime-pull"}],
    }
    if legacy:
        values["env"].append({"name": "TRAINING_RAY_IMAGE", "value": image})
    else:
        values["trainingRuntime"] = {"image": image}
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump(values))
    rendered = deploy.command(["helm", "template", "e2e", deploy.CHART, "-f", path])
    deployment = next(doc for doc in yaml.safe_load_all(rendered) if doc["kind"] == "Deployment")
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    assert [entry for entry in env if entry["name"] == "TRAINING_RAY_IMAGE"] == [
        {"name": "TRAINING_RAY_IMAGE", "value": image}
    ]
    assert {"name": "UNRELATED_SETTING", "value": "preserved"} in env
    pod = deployment["spec"]["template"]["spec"]
    init = pod["initContainers"][0]
    assert init["name"] == "discover-algorithms"
    assert init["image"] == image
    assert init["imagePullPolicy"] == "Always"
    assert init["command"][2] == image
    assert init["command"][3] == "/var/run/training-service/catalog/catalog.json"
    assert not any("gpu" in key for key in init["resources"]["limits"])
    assert pod["imagePullSecrets"] == [{"name": "private-runtime-pull"}]
    assert {"name": "TRAINING_SERVICE_CATALOG_PATH", "value": init["command"][3]} in env
    assert not any(
        e["name"] in {"TRAINING_SERVICE_CATALOG_NAME", "TRAINING_SERVICE_CATALOG_NAMESPACE"}
        for e in env
    )
    mounted = next(
        m for m in pod["containers"][0]["volumeMounts"] if m["name"] == "training-algorithm-catalog"
    )
    assert mounted["name"] == "training-algorithm-catalog" and mounted["readOnly"] is True
    exporter = next(d for d in yaml.safe_load_all(rendered) if d["kind"] == "ConfigMap")
    script = deploy.CHART / "files/export_algorithm_catalog.py"
    assert exporter["data"]["export_algorithm_catalog.py"] == script.read_text()


def test_chart_rejects_duplicate_training_runtime_image(tmp_path):
    import shutil

    if not shutil.which("helm"):
        pytest.skip("helm is not installed")
    path = tmp_path / "values.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "trainingRuntime": {"image": "dedicated-image"},
                "env": [{"name": "TRAINING_RAY_IMAGE", "value": "legacy-image"}],
            }
        )
    )
    rendered = subprocess.run(
        ["helm", "template", "e2e", str(deploy.CHART), "-f", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rendered.returncode != 0
    assert "trainingRuntime.image" in rendered.stderr
    assert "env" in rendered.stderr


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


def stream_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        training_jobs,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        ),
    )
    return now


def test_stream_reconnects_after_quiet_read_and_preserves_incremental_events(monkeypatch):
    now = stream_clock(monkeypatch)
    calls = []

    class QuietStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'event: log\ndata: {"message":"first"}\n\n'
            now[0] += 15
            raise httpx.ReadTimeout("No logs during model loading")

    def transport(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, stream=QuietStream()
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=b'event: log\ndata: {"message":"second"}\n\n',
        )

    with httpx.Client(
        base_url="https://unit.invalid",
        headers={"Authorization": "Bearer unit-token"},
        transport=httpx.MockTransport(transport),
    ) as http:
        assert follow_logs(LiveAPI(http, "test-project"), "/jobs/e2e-owned", 60) == 2
    assert len(calls) == 2


def test_stream_quiet_reads_exhaust_only_the_overall_deadline(monkeypatch):
    now = stream_clock(monkeypatch)
    read_timeouts = []

    def transport(request):
        read_timeout = request.extensions["timeout"]["read"]
        read_timeouts.append(read_timeout)
        now[0] += read_timeout
        raise httpx.ReadTimeout("No new log events")

    with httpx.Client(
        base_url="https://unit.invalid", transport=httpx.MockTransport(transport)
    ) as http:
        with pytest.raises(pytest.fail.Exception, match="SSE deadline exceeded"):
            follow_logs(LiveAPI(http, "test-project"), "/jobs/e2e-owned", 60)
    assert read_timeouts == [15, 15, 15, 9]
    assert now[0] == 60


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ConnectTimeout])
def test_stream_does_not_retry_other_transport_errors(monkeypatch, failure):
    stream_clock(monkeypatch)
    transport = Mock(side_effect=failure("Cannot connect"))
    with httpx.Client(
        base_url="https://unit.invalid", transport=httpx.MockTransport(transport)
    ) as http:
        with pytest.raises(pytest.fail.Exception, match="SSE connection/read failed"):
            follow_logs(LiveAPI(http, "test-project"), "/jobs/e2e-owned", 60)
    assert transport.call_count == 1


@pytest.mark.parametrize("status", [401, 403, 500])
def test_stream_does_not_retry_http_failures(monkeypatch, status):
    stream_clock(monkeypatch)
    transport = Mock(return_value=httpx.Response(status))
    with httpx.Client(
        base_url="https://unit.invalid", transport=httpx.MockTransport(transport)
    ) as http:
        with pytest.raises(pytest.fail.Exception, match=f"Log stream returned HTTP {status}"):
            follow_logs(LiveAPI(http, "test-project"), "/jobs/e2e-owned", 60)
    assert transport.call_count == 1


def test_explicit_api_token_takes_precedence_over_oc_session(monkeypatch):
    monkeypatch.setenv("TRAINING_SERVICE_TOKEN", "explicit-test-token")
    run = Mock()
    monkeypatch.setattr(support.subprocess, "run", run)
    assert support.api_token() == "explicit-test-token"
    run.assert_not_called()


@pytest.mark.parametrize("configured", [None, "", "  "])
def test_api_token_defaults_to_current_oc_session(monkeypatch, capsys, configured):
    if configured is None:
        monkeypatch.delenv("TRAINING_SERVICE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TRAINING_SERVICE_TOKEN", configured)
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "session-test-token\n", ""))
    monkeypatch.setattr(support.subprocess, "run", run)
    assert support.api_token() == "session-test-token"
    assert run.call_args.args[0] == ["oc", "whoami", "-t"]
    assert run.call_args.kwargs["capture_output"] is True
    assert run.call_args.kwargs["timeout"] == 15
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "result",
    [
        subprocess.CompletedProcess([], 1, "private-output", "private-output"),
        subprocess.CompletedProcess([], 0, "", "private-output"),
        subprocess.CompletedProcess([], 0, "private-output\nnot-a-token", ""),
        FileNotFoundError("private-output"),
        subprocess.TimeoutExpired("oc", 15, output="private-output"),
    ],
)
def test_oc_token_failures_are_actionable_and_redacted(monkeypatch, capsys, result):
    monkeypatch.delenv("TRAINING_SERVICE_TOKEN", raising=False)
    run = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr(support.subprocess, "run", run)
    with pytest.raises(ValueError, match="oc login") as error:
        support.api_token()
    assert "TRAINING_SERVICE_TOKEN" in str(error.value)
    assert "private-output" not in str(error.value)
    assert capsys.readouterr() == ("", "")


def test_existing_deployment_config_resolves_default_api_token(monkeypatch):
    monkeypatch.setenv("TRAINING_SERVICE_URL", "https://unit.invalid")
    monkeypatch.setenv("TRAINING_SERVICE_PROJECT", "unit-project")
    monkeypatch.delenv("TRAINING_SERVICE_TOKEN", raising=False)
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "session-test-token\n", ""))
    monkeypatch.setattr(support.subprocess, "run", run)
    config = Mock()
    config.option.collectonly = False
    config.getoption.side_effect = lambda option: (
        30
        if option in {"job_timeout", "cleanup_timeout", "request_timeout", "stream_timeout"}
        else False
    )
    config.getini.return_value = False
    live_config.pytest_configure(config)
    assert os.environ["TRAINING_SERVICE_TOKEN"] == "session-test-token"


def test_runner_missing_oc_credential_fails_before_build(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAINING_SERVICE_TOKEN", raising=False)
    monkeypatch.setattr(
        support.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 1, "", "not logged in")),
    )
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
