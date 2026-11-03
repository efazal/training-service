"""Configuration and bounded cleanup for tests against an existing deployment."""

from __future__ import annotations

import json
import os
import ssl
import time
import uuid
from pathlib import Path

import httpx
import pytest

from tests.e2e.support import PROFILES, namespace, workload

TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
STATUSES = TERMINAL | {"PENDING", "RUNNING"}
POLL_SECONDS = 5
OUTCOMES = {"passed": 0, "failed": 0, "skipped": 0}


def pytest_addoption(parser):
    group = parser.getgroup("live Training Service")
    group.addoption("--job-file", help="Opt into real training using this JSON request file.")
    group.addoption("--job-timeout", type=int, default=1800, help="Training deadline in seconds.")
    group.addoption("--cleanup-timeout", type=int, default=180, help="Cleanup deadline in seconds.")
    group.addoption("--request-timeout", type=int, default=30, help="HTTP timeout in seconds.")
    group.addoption("--check-logs", action="store_true", help="Require a non-empty log snapshot.")
    group.addoption(
        "--check-stream", action="store_true", help="Require incremental SSE driver logs."
    )
    group.addoption("--stream-timeout", type=int, default=60)
    group.addoption("--queue-profile", choices=PROFILES)
    group.addoption(
        "--observe-cluster",
        action="store_true",
        help="Verify owner-scoped GPU cleanup using kubeconfig.",
    )
    group.addoption("--artifact-pod", help="Prepared pod with read-only output PVC mount.")
    group.addoption("--artifact-container", default="verifier")
    group.addoption(
        "--keep-failed-job", action="store_true", help="Retain terminal failed jobs for debugging."
    )
    group.addoption("--forbidden-project", help="Existing project this caller must not access.")
    group.addoption(
        "--missing-submission-job",
        help="Existing terminal RayJob whose live cluster no longer has its Ray submission.",
    )
    group.addoption(
        "--expect-invalid-kubernetes-url",
        action="store_true",
        help="Opt into checks against an isolated deployment with an invalid Kubernetes URL.",
    )


def pytest_configure(config):
    OUTCOMES.update(passed=0, failed=0, skipped=0)
    if config.option.collectonly:
        return
    missing = [
        name
        for name in ("TRAINING_SERVICE_URL", "TRAINING_SERVICE_PROJECT", "TRAINING_SERVICE_TOKEN")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise pytest.UsageError("Live E2E tests require: " + ", ".join(missing))
    for option in ("job_timeout", "cleanup_timeout", "request_timeout", "stream_timeout"):
        if config.getoption(option) <= 0:
            raise pytest.UsageError(f"--{option.replace('_', '-')} must be positive")
    for option in (
        "check_logs",
        "check_stream",
        "observe_cluster",
        "artifact_pod",
        "queue_profile",
    ):
        if config.getoption(option) and not config.getoption("job_file"):
            raise pytest.UsageError(f"--{option.replace('_', '-')} requires --job-file")
    if config.getoption("job_file"):
        try:
            workload(config.getoption("job_file"), config.getoption("queue_profile"))
        except ValueError as error:
            raise pytest.UsageError(str(error)) from None
        if config.getoption("queue_profile") in {"with-kueue", "no-kueue"} and not config.getoption(
            "observe_cluster"
        ):
            raise pytest.UsageError("Kueue enabled/absent proof requires --observe-cluster")
    if config.getoption("artifact_pod") and not config.getoption("observe_cluster"):
        raise pytest.UsageError("--artifact-pod requires --observe-cluster")
    if config.getoption("showlocals") or config.getini("log_cli"):
        raise pytest.UsageError(
            "Do not enable --showlocals or live debug logging with real credentials"
        )


def pytest_runtest_logreport(report):
    if report.when == "call" or report.failed or report.skipped:
        OUTCOMES[report.outcome] += 1


def pytest_sessionfinish(session, exitstatus):
    if not session.config.option.collectonly and exitstatus == 0 and not OUTCOMES["passed"]:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    directory = os.getenv("TRAINING_SERVICE_EVIDENCE_DIR")
    if directory and not session.config.option.collectonly:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tests.json").write_text(json.dumps(OUTCOMES, indent=2) + "\n")


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        "Sanity result only, not full customer-readiness certification. "
        "See tests/e2e/coverage.md for deferred/conditional cases. "
        f"Skipped cases: {OUTCOMES['skipped']}."
    )


def _project(value):
    try:
        return namespace(value)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from None


class LiveAPI:
    """HTTP-only test client; failure messages never dump headers or response bodies."""

    def __init__(self, client, project):
        self.client = client
        self.project = project
        self.jobs = f"/api/v1/projects/{project}/training-jobs"

    def request(self, method, path, expected=200, authenticated=True, **kwargs):
        try:
            request = self.client.build_request(method, path, **kwargs)
            if not authenticated:
                request.headers.pop("Authorization", None)
            response = self.client.send(request)
        except httpx.HTTPError as error:
            pytest.fail(f"{method} {path}: {type(error).__name__}", pytrace=False)
        token = request.headers.get("Authorization", "").partition(" ")[2]
        if token and token in response.text:
            pytest.fail("Response exposed the request credential", pytrace=False)
        if response.status_code == 401:
            challenge = response.headers.get("WWW-Authenticate", "")
            assert challenge.partition(" ")[0].lower() == "bearer", (
                "HTTP 401 must include WWW-Authenticate: Bearer"
            )
        allowed = (expected,) if isinstance(expected, int) else expected
        if response.status_code not in allowed:
            pytest.fail(
                f"{method} {path}: expected HTTP {allowed}, got {response.status_code}",
                pytrace=False,
            )
        return response

    def object(self, response):
        try:
            data = response.json()
        except ValueError:
            pytest.fail("Expected a JSON response", pytrace=False)
        if not isinstance(data, dict):
            pytest.fail("Expected a JSON object response", pytrace=False)
        return data

    def error(self, response, code, *, mentions=()):
        assert response.headers.get("content-type", "").startswith("application/json")
        data = self.object(response)
        assert set(data) == {"code", "message"}
        assert data.get("code") == code
        assert isinstance(data.get("message"), str) and data["message"]
        for fragment in mentions:
            assert fragment in data["message"], "Error message is missing resource context"

    def summary(self, item):
        assert {"id", "status", "created_at"} <= item.keys()
        assert item.keys() <= {"id", "status", "created_at", "queue_name"}
        assert isinstance(item["id"], str) and item["id"]
        assert item["status"] in STATUSES
        assert isinstance(item["created_at"], str) and item["created_at"]

    def page(self, params):
        data = self.object(self.request("GET", self.jobs, params=params))
        assert isinstance(data.get("items"), list)
        assert type(data.get("total_count")) is int
        assert data["total_count"] >= len(data["items"])
        assert len(data["items"]) <= params["limit"]
        for item in data["items"]:
            self.summary(item)
        return data

    def find_job(self, job_id, **filters):
        params = {"limit": 20, **filters}
        tokens = set()
        for _ in range(100):
            page = self.page(params)
            for item in page["items"]:
                if "status" in filters:
                    assert item["status"] == filters["status"]
                if item["id"] == job_id:
                    return item
            token = page.get("next_page_token")
            assert token and token not in tokens, "Created job missing from project listing"
            tokens.add(token)
            params["page_token"] = token
        pytest.fail("Created job not found within 100 list pages")

    def cleanup(self, job_id, timeout, expected_submission):
        """Delete only this run's terminal job; never cancel active training."""
        path = f"{self.jobs}/{job_id}"
        deadline = time.monotonic() + timeout
        deleting = False
        while True:
            response = self.request("GET", path, expected=(200, 404))
            if response.status_code == 404:
                return
            detail = self.object(response)
            if detail.get("submission") != expected_submission:
                pytest.fail(
                    "Cleanup refused: job submission does not match this test's fixture",
                    pytrace=False,
                )
            status = detail.get("status")
            if not deleting and status in TERMINAL:
                deleted = self.request("DELETE", path, expected=(204, 404, 409))
                deleting = deleted.status_code in {204, 404}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(
                    f"Cleanup incomplete for {self.project}/{job_id} (status={status}). "
                    "Only terminal jobs can be deleted; inspect this job manually.",
                    pytrace=False,
                )
            time.sleep(min(POLL_SECONDS, remaining))


@pytest.fixture(scope="session")
def api(pytestconfig):
    project = _project(os.environ["TRAINING_SERVICE_PROJECT"].strip())
    try:
        origin = httpx.URL(os.environ["TRAINING_SERVICE_URL"].strip())
    except httpx.InvalidURL:
        raise pytest.UsageError("TRAINING_SERVICE_URL must be a service origin URL") from None
    if (
        not origin.host
        or origin.userinfo
        or origin.query
        or origin.fragment
        or origin.path != "/"
        or origin.scheme not in {"http", "https"}
    ):
        raise pytest.UsageError("Use the service origin, without credentials or /api/v1")
    if origin.scheme == "http" and origin.host not in {"localhost", "127.0.0.1", "::1"}:
        raise pytest.UsageError("Bearer tokens require HTTPS, except for loopback port-forwarding")
    ca_bundle = os.getenv("TRAINING_SERVICE_CA_BUNDLE") or None
    try:
        tls = ssl.create_default_context(cafile=ca_bundle)
    except (OSError, ssl.SSLError):
        raise pytest.UsageError("Cannot load TRAINING_SERVICE_CA_BUNDLE") from None
    with httpx.Client(
        base_url=origin,
        headers={"Authorization": "Bearer " + os.environ["TRAINING_SERVICE_TOKEN"].strip()},
        verify=tls,
        timeout=httpx.Timeout(pytestconfig.getoption("request_timeout"), connect=5),
        follow_redirects=False,
        trust_env=False,
    ) as client:
        yield LiveAPI(client, project)


@pytest.fixture
def missing_job_id():
    return "e2e-missing-" + uuid.uuid4().hex


@pytest.fixture
def forbidden_project(pytestconfig):
    project = pytestconfig.getoption("forbidden_project")
    if not project:
        pytest.skip("Pass --forbidden-project to check a caller-inaccessible existing project")
    return _project(project)


@pytest.fixture
def training_job(api, pytestconfig):
    filename = pytestconfig.getoption("job_file")
    if not filename:
        pytest.skip("Real training is opt-in: pass --job-file /path/to/request.json")
    payload = workload(filename, pytestconfig.getoption("queue_profile"))
    # RHOAI adds cluster suffixes and an auth-volume prefix with a 63-char limit.
    job_id = "e2e-" + uuid.uuid4().hex[:16]
    payload["name"] = job_id
    if "output" in payload:
        output = payload["output"]
        if not isinstance(output, dict) or not isinstance(output.get("destination_uri"), str):
            raise pytest.UsageError("Job output must have a destination_uri string")
        output["destination_uri"] = output["destination_uri"].rstrip("/") + f"/{job_id}/"
    print(f"E2E job: {api.project}/{job_id}")
    api.request("GET", f"{api.jobs}/{job_id}", expected=404)
    # Register cleanup before POST: a lost response may still have created the job.
    try:
        yield job_id, payload
    finally:
        retained = False
        if pytestconfig.getoption("keep_failed_job"):
            response = api.request("GET", f"{api.jobs}/{job_id}", expected=(200, 404))
            if response.status_code == 200:
                retained = api.object(response).get("status") in {"FAILED", "STOPPED"}
        if retained:
            print(f"Retained terminal failed job for inspection: {api.project}/{job_id}")
        else:
            api.cleanup(job_id, pytestconfig.getoption("cleanup_timeout"), payload)


@pytest.fixture
def observer(api, pytestconfig):
    if not pytestconfig.getoption("observe_cluster"):
        yield None
        return
    from tests.e2e.observer import Observer

    try:
        result = Observer(api.project)
        payload = workload(
            pytestconfig.getoption("job_file"), pytestconfig.getoption("queue_profile")
        )
        result.preflight(payload, pytestconfig.getoption("queue_profile"))
        if pod := pytestconfig.getoption("artifact_pod"):
            result.artifact_mount(pod, pytestconfig.getoption("artifact_container"), payload)
    except Exception:
        pytest.fail(
            "Observer preflight failed: check queue/profile, kubeconfig, permissions and fixtures",
            pytrace=False,
        )
    try:
        yield result
    finally:
        result.api.close()
