"""Exercise the pinned Ray transport adapter using dummy credentials on loopback."""

import asyncio
import ssl
from unittest.mock import Mock

import pytest
from aiohttp import web

from training_service import logs
from training_service.errors import ApiError
from training_service.logs import BoundedJobClient, CodeFlareLogClient


def test_missing_version_endpoint_is_not_reported_as_logs_not_ready(monkeypatch):
    from ray.dashboard.modules import dashboard_sdk

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_API_SERVER_ADDRESS", raising=False)
    transport = Mock(return_value=Mock(status_code=404, text="sensitive-gateway-response"))
    monkeypatch.setattr(dashboard_sdk.requests, "request", transport)

    with pytest.raises(ApiError) as raised:
        BoundedJobClient(
            "https://ray.example.com/ray/team-a/example-cluster",
            headers={"Authorization": "Bearer dummy-unit-test-token"},
        )

    assert transport.call_count == 1
    assert transport.call_args.args[1].endswith("/api/version")
    assert raised.value.status_code == 503
    assert raised.value.code == "logs_unavailable"
    assert "sensitive-gateway-response" not in raised.value.message
    assert "ray.example.com" not in raised.value.message
    assert "dummy-unit-test-token" not in raised.value.message


@pytest.mark.parametrize(
    "endpoint, status, code",
    [
        ("/api/version", 503, "logs_unavailable"),
        ("/api/jobs/", 503, "logs_unavailable"),
        ("/api/jobs/raysubmit_123", 409, "logs_not_ready"),
        ("/api/jobs/raysubmit_123/logs", 409, "logs_not_ready"),
        ("/api/jobs/raysubmit_123/unknown", 503, "logs_unavailable"),
    ],
)
def test_log_http_404_only_reports_readiness_for_job_reads(monkeypatch, endpoint, status, code):
    from ray.dashboard.modules import dashboard_sdk

    transport = Mock(return_value=Mock(status_code=404))
    monkeypatch.setattr(dashboard_sdk.requests, "request", transport)
    sdk = object.__new__(BoundedJobClient)
    sdk._address = "https://ray.example.com"
    sdk._cookies = None
    sdk._headers = {"Authorization": "Bearer dummy-unit-test-token"}
    sdk._verify = True

    with pytest.raises(ApiError) as raised:
        sdk._do_request("GET", endpoint)

    assert raised.value.status_code == status
    assert raised.value.code == code


@pytest.mark.parametrize("redirect", [False, True])
def test_log_websocket_streams_normally_but_never_follows_redirects(redirect):
    async def exercise():
        seen = []

        async def endpoint(request):
            seen.append((request.path, request.headers.get("Authorization")))
            if redirect:
                raise web.HTTPFound("/different-path")
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            await socket.send_str("one log line\n")
            await socket.close()
            return socket

        async def redirected(request):
            seen.append((request.path, request.headers.get("Authorization")))
            return web.Response()

        application = web.Application()
        application.router.add_get("/api/jobs/example/logs/tail", endpoint)
        application.router.add_get("/different-path", redirected)
        runner = web.AppRunner(application)
        await runner.setup()
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            sdk = object.__new__(BoundedJobClient)
            sdk._address = f"http://127.0.0.1:{runner.addresses[0][1]}"
            sdk._headers = {"Authorization": "Bearer dummy-unit-test-token"}
            sdk._cookies = None
            sdk._ssl_context = ssl.create_default_context()
            if redirect:
                with pytest.raises(ApiError) as raised:
                    _ = [chunk async for chunk in sdk.tail_job_logs("example")]
                assert raised.value.status_code == 503
            else:
                assert [chunk async for chunk in sdk.tail_job_logs("example")] == ["one log line\n"]
            assert seen == [("/api/jobs/example/logs/tail", "Bearer dummy-unit-test-token")]
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


def test_codeflare_log_methods_delegate_to_hardened_transport(monkeypatch):
    transport = Mock()
    factory = Mock(return_value=transport)
    monkeypatch.setattr(logs, "BoundedJobClient", factory)
    sdk = CodeFlareLogClient(
        "https://ray.example.com", headers={"Authorization": "Bearer dummy-unit-test-token"}
    )
    assert sdk.get_job_logs("example") is transport.get_job_logs.return_value
    assert sdk.get_job_status("example") is transport.get_job_status.return_value
    assert sdk.tail_job_logs("example") is transport.tail_job_logs.return_value
    factory.assert_called_once_with(
        "https://ray.example.com",
        headers={"Authorization": "Bearer dummy-unit-test-token"},
        verify=True,
    )
    transport.get_job_logs.assert_called_once_with(job_id="example")
    transport.get_job_status.assert_called_once_with(job_id="example")
    transport.tail_job_logs.assert_called_once_with(job_id="example")
