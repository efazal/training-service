"""Adversarial checks for the live log-timeout assertions; no deployment is contacted."""

import json
from types import SimpleNamespace

import httpx
import pytest
from tests.e2e import test_training_jobs as training_jobs
from tests.e2e.conftest import LiveAPI
from tests.e2e.test_training_jobs import verify_stream_timeout


@pytest.fixture
def timeout_api(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        training_jobs,
        "time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )
    clients = []

    def build(fault=None, limit=None):
        streams = []
        details = []

        class Stream(httpx.AsyncByteStream):
            def __init__(self, expected, number):
                self.expected = expected
                self.number = number
                self.closed = False

            async def __aiter__(self):
                if fault != "empty_reconnect" or self.number == 1:
                    yield b'event: log\ndata: {"message":"training continues"}\n\n'
                if fault not in {"immediate", "expiry_before_close"}:
                    now[0] += 1 if fault == "fixed_timeout" else self.expected
                if fault == "late":
                    now[0] += 2
                event = "log" if fault == "wrong_event" else "error"
                data = {"code": "stream_expired", "message": "Reconnect to continue."}
                frame = f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}"
                yield (frame + ("\n" if fault == "truncated" else "\n\n")).encode()
                if fault == "expiry_before_close":
                    now[0] += self.expected
                if fault == "after_expiry":
                    yield b'event: log\ndata: {"message":"unexpected later event"}\n\n'

            async def aclose(self):
                self.closed = True

        def sync_transport(request):
            if request.url.path.endswith("/logs"):
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/plain"},
                    content="training continues",
                )
            details.append(request)
            status = "STOPPED" if fault == "stopped" and len(details) > 1 else "RUNNING"
            return httpx.Response(200, json={"status": status})

        async def async_transport(request):
            requested = request.url.params.get("timeout_seconds")
            expected = min(int(requested), limit or 300) if requested else (limit or 300)
            if fault == "ignored_cap" and requested:
                expected = int(requested)
            if fault == "wrong_default" and requested is None:
                expected = 1
            stream = Stream(expected, len(streams) + 1)
            streams.append((request, stream))
            if fault == "reconnect_denied" and len(streams) == 2:
                return httpx.Response(403, json={"code": "forbidden", "message": "Denied"})
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

        http = httpx.Client(
            base_url="https://unit.invalid",
            headers={"Authorization": "Bearer unit-token"},
            transport=httpx.MockTransport(sync_transport),
        )
        clients.append(http)
        api = LiveAPI(
            http,
            "test-project",
            async_client=lambda: httpx.AsyncClient(
                base_url=http.base_url,
                headers=http.headers,
                transport=httpx.MockTransport(async_transport),
            ),
        )
        api.log_stream_max_seconds = limit
        return api, streams, details

    yield build
    for client in clients:
        client.close()


@pytest.mark.parametrize(
    "fault",
    [
        "immediate",
        "fixed_timeout",
        "late",
        "wrong_event",
        "truncated",
        "after_expiry",
        "empty_reconnect",
        "reconnect_denied",
        "stopped",
        "expiry_before_close",
    ],
)
def test_timeout_verification_rejects_false_passes(timeout_api, fault):
    api, _, _ = timeout_api(fault=fault)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        verify_stream_timeout(api, "/jobs/e2e-owned")
    assert not api.client.is_closed


@pytest.mark.parametrize("limit", [None, 1, 3])
def test_timeout_verification_measures_requests_reconnects_and_checks_server_cap(
    timeout_api, limit
):
    api, streams, details = timeout_api(limit=limit)
    results = verify_stream_timeout(api, "/jobs/e2e-owned")
    requests = [1, 3] + ([None, 2147483647] if limit is not None else [])
    assert [result["requested_seconds"] for result in results] == requests
    assert [result["expected_seconds"] for result in results] == [
        min(value, limit or 300) if value is not None else limit for value in requests
    ]
    assert all(result["elapsed_seconds"] == result["expected_seconds"] for result in results)
    assert len(details) == len(requests) + 1
    assert len(streams) == len(requests)
    assert all(stream.closed for _, stream in streams)
    assert not api.client.is_closed


@pytest.mark.parametrize("fault", ["ignored_cap", "wrong_default"])
def test_timeout_verification_rejects_ignored_deployment_limit(timeout_api, fault):
    api, _, _ = timeout_api(fault=fault, limit=3)
    with pytest.raises(AssertionError, match="Expected expiry after 3s"):
        verify_stream_timeout(api, "/jobs/e2e-owned")
