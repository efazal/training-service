"""Credential-free runner/observer safety tests, not substitutes for live E2E."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from kubernetes.client.exceptions import ApiException
from tests.e2e import deploy
from tests.e2e.conftest import LiveAPI
from tests.e2e.observer import Observer, owned_by, pages
from tests.e2e.support import pvc_path, workload
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


def test_subprocess_errors_withhold_raw_output(monkeypatch):
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
        deploy.command(["helm", "install", "test"])
    assert "private" not in str(error.value)
    assert "exit 1" in str(error.value)


@pytest.mark.parametrize("failure,keep", [(False, False), (False, True), (True, False)])
def test_runner_build_push_digest_install_test_cleanup(monkeypatch, tmp_path, failure, keep):
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
        ]
    )
    monkeypatch.setattr(deploy, "preflight", Mock())
    monkeypatch.setattr(deploy, "wait_ready", Mock())
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
        if "pytest" in command and failure:
            raise RuntimeError("live tests failed")
        return ""

    monkeypatch.setattr(deploy, "command", run)
    if failure:
        with pytest.raises(RuntimeError, match="live tests failed"):
            deploy.run(args)
    else:
        deploy.run(args)
    build = next(c for c in commands if c[:2] == ["podman", "build"])
    assert "Containerfile" in build and "Dockerfile" not in build
    install = next(c for c in commands if c[:2] == ["helm", "install"])
    assert any(c.startswith("image.tag=e2e-") and c.endswith("@" + digest) for c in install)
    assert install[2].startswith("training-e2e-") and "upgrade" not in install
    assert any("pytest" in c for c in commands)
    assert any(c[:2] == ["helm", "uninstall"] for c in commands) == (not failure and not keep)
    report = json.loads(next((tmp_path / "results").glob("*/run.json")).read_text())
    assert report["outcome"] == ("incomplete" if failure else "passed")
    assert report["image"].endswith("@" + digest)


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
