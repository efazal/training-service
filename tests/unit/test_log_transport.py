"""Exercise the pinned Ray transport adapter using dummy credentials on loopback."""

import asyncio
import ssl

import pytest
from aiohttp import web

from training_service.errors import ApiError
from training_service.ray_logs import BoundedJobClient


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
