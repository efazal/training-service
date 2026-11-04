"""One isolated, real training run from submission through terminal deletion."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
import pytest

TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}


def follow_logs(api, path, timeout):
    """Consume real SSE while training runs, close the connection after two chunks."""
    deadline = time.monotonic() + timeout
    chunks = set()
    while time.monotonic() < deadline:
        try:
            with api.client.stream(
                "GET", path + "/logs", params={"follow": "true"}, timeout=min(15, timeout)
            ) as response:
                if response.status_code == 409:
                    response.read()
                    api.error(response, "logs_not_ready")
                elif response.status_code != 200:
                    pytest.fail(f"Log stream returned HTTP {response.status_code}", pytrace=False)
                else:
                    assert response.headers.get("content-type", "").startswith("text/event-stream")
                    event, data = None, []
                    for line in response.iter_lines():
                        if time.monotonic() >= deadline:
                            break
                        token = api.client.headers["Authorization"].partition(" ")[2]
                        if token in line:
                            pytest.fail("Log stream exposed caller credential", pytrace=False)
                        if line.startswith("event: "):
                            event = line[7:]
                        elif line.startswith("data: "):
                            data.append(line[6:])
                        elif not line and data:
                            body = json.loads("\n".join(data))
                            assert event == "log", "Unexpected SSE error/non-log event"
                            assert isinstance(body.get("message"), str)
                            if body["message"].strip():
                                chunks.add(hashlib.sha256(body["message"].encode()).hexdigest())
                            if len(chunks) >= 2:
                                return len(chunks)
                            event, data = None, []
                    pytest.fail(
                        "Stream ended without two distinct incremental log events", pytrace=False
                    )
        except httpx.HTTPError:
            pytest.fail(
                "SSE connection/read failed before incremental logs were verified", pytrace=False
            )
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    pytest.fail("SSE deadline exceeded before incremental log events arrived", pytrace=False)


def observe(observer, job_id):
    if observer:
        try:
            observer.observe(job_id)
        except Exception as error:
            from tests.e2e.observer import failure_reason

            pytest.fail("Cluster observation failed: " + failure_reason(error), pytrace=False)


def test_training_job_lifecycle(api, training_job, pytestconfig, observer, request):
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            lifecycle(api, training_job, pytestconfig, observer, executor)
    except (AssertionError, pytest.fail.Exception):
        if observer and observer.queue:
            request.node.user_properties.append(("observer_failure", observer.queue_diagnostic()))
        raise


def lifecycle(api, training_job, pytestconfig, observer, executor):
    job_id, payload = training_job
    path = f"{api.jobs}/{job_id}"
    response = api.request("POST", api.jobs, expected=202, json=payload)
    receipt = api.object(response)
    assert receipt["id"] == job_id
    assert receipt["name"] == job_id
    assert receipt["project"] == api.project
    assert receipt["status"] == "SUBMITTED"
    assert receipt["selected_backend"] == payload.get("backend", "ray")
    # Never follow a server-supplied URL with the bearer token.
    assert response.headers.get("location") == path

    detail = api.object(api.request("GET", path))
    observe(observer, job_id)
    assert detail["id"] == job_id
    assert detail["project"] == api.project
    assert isinstance(detail.get("submitted_by"), str) and detail["submitted_by"]
    assert detail.get("submission") == payload
    assert detail.get("created_at")
    assert detail.get("selected_backend") == payload.get("backend", "ray")
    if "queue_name" in payload:
        assert detail.get("queue_name") == payload["queue_name"]
    if "resources" in payload:
        assert detail.get("resources")
        for key, value in payload["resources"].items():
            assert detail["resources"].get(key) == value, (
                "Observed resources differ from submission"
            )

    duplicate = api.request("POST", api.jobs, expected=409, json=payload)
    api.error(
        duplicate,
        "already_exists",
        mentions=(f"Training job '{job_id}'", f"project '{api.project}'", "already exists"),
    )

    api.find_job(job_id)

    deadline = time.monotonic() + pytestconfig.getoption("job_timeout")
    check_logs = pytestconfig.getoption("check_logs") or pytestconfig.getoption("check_stream")
    stream_required = pytestconfig.getoption("check_stream")
    stream = None
    saw_logs = False
    last_log_status = None
    previous = None
    while True:
        detail = api.object(api.request("GET", path))
        observe(observer, job_id)
        if stream and stream.done():
            stream.result()  # Fail immediately on a real SSE failure.
        status = detail["status"]
        assert status in TERMINAL | {"PENDING", "RUNNING"}
        if status != previous:
            print(f"E2E job {job_id}: {status}")
            previous = status
        if check_logs and not saw_logs:
            logs = api.request(
                "GET",
                path + "/logs",
                expected=(200, 404, 409, 410, 503),
                params={"follow": "false", "tail_lines": 20},
            )
            last_log_status = logs.status_code
            if logs.status_code == 200:
                assert "text/plain" in logs.headers.get("content-type", "")
                saw_logs = bool(logs.text.strip())
            elif logs.status_code == 409:
                api.error(logs, "logs_not_ready")
            elif logs.status_code == 404:
                api.error(logs, "not_found")
                pytest.fail(
                    f"Ray submission missing for {api.project}/{job_id}; not retrying log reads.",
                    pytrace=False,
                )
            elif logs.status_code == 410:
                api.error(logs, "ray_cluster_gone")
            else:
                api.error(logs, "logs_unavailable")
        # Scheduling/image pulls use the training deadline, not the SSE deadline.
        if saw_logs and stream_required and stream is None and status not in TERMINAL:
            stream = executor.submit(
                follow_logs, api, path, pytestconfig.getoption("stream_timeout")
            )
        if status in TERMINAL:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            queue = observer.queue_diagnostic() if observer else ""
            pytest.fail(
                f"Training timed out: {api.project}/{job_id} is {status}. {queue}", pytrace=False
            )
        time.sleep(min(5, remaining))

    assert status == "SUCCEEDED", f"Training ended as {status}: inspect {api.project}/{job_id}"
    assert detail.get("submission") == payload
    timestamps = [
        datetime.fromisoformat(detail[key].replace("Z", "+00:00"))
        for key in ("created_at", "started_at", "finished_at")
        if key in detail
    ]
    assert all(timestamp.tzinfo is not None for timestamp in timestamps)
    assert timestamps == sorted(timestamps)
    if check_logs:
        assert saw_logs, f"No non-empty driver log snapshot; last HTTP status: {last_log_status}"
    if stream_required:
        assert stream is not None, "No live streaming window was observed before training completed"
        assert stream.result() >= 2

    artifact_result = None
    if pod := pytestconfig.getoption("artifact_pod"):
        try:
            artifact_result = observer.artifacts(
                pod, pytestconfig.getoption("artifact_container"), payload
            )
        except Exception:
            pytest.fail(
                "LoRA checkpoint verification failed; inspect retained output directory",
                pytrace=False,
            )

    # A terminal job must be included with and without the explicit status filter.
    assert api.find_job(job_id)["status"] == "SUCCEEDED"
    assert api.find_job(job_id, status="SUCCEEDED")["status"] == "SUCCEEDED"
    api.cleanup(job_id, pytestconfig.getoption("cleanup_timeout"), payload)
    if observer:
        try:
            observer.verify_cleanup(job_id, pytestconfig.getoption("cleanup_timeout"))
            if artifact_result:
                observer.artifacts(pod, pytestconfig.getoption("artifact_container"), payload)
        except Exception:
            pytest.fail(
                "Independent resource/GPU cleanup or output preservation proof failed",
                pytrace=False,
            )
    for method in ("GET", "DELETE"):
        api.error(
            api.request(method, path, expected=404),
            "not_found",
            mentions=(f"Training job '{job_id}'", f"project '{api.project}'", "not found"),
        )
    if directory := os.getenv("TRAINING_SERVICE_EVIDENCE_DIR"):
        evidence = {
            "job": job_id,
            "profile": pytestconfig.getoption("queue_profile"),
            "succeeded": True,
            "snapshot_logs": saw_logs,
            "sse": stream is not None,
            "gpu_cleanup": bool(observer and observer.saw_gpu),
            "queue_admitted": bool(observer and observer.admitted),
            "artifacts": artifact_result or "not verified: no --artifact-pod selected",
        }
        (Path(directory) / "lifecycle.json").write_text(json.dumps(evidence, indent=2) + "\n")
