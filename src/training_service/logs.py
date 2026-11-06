"""Backend-neutral driver log responses and the current CodeFlare transport."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Protocol, cast

import aiohttp
import anyio
from codeflare_sdk.ray.client.ray_jobs import RayJobClient
from fastapi import Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from ray.exceptions import AuthenticationError
from ray.job_submission import JobSubmissionClient

from training_service.errors import ApiError

if TYPE_CHECKING:
    from training_service.jobs import JobsBackend


def log_stream_timeout_seconds() -> int:
    """Read the deployment's positive stream lifetime once at application creation."""
    setting = "TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS"
    try:
        seconds = int(os.getenv(setting, "300"))
        if seconds > 0:
            return seconds
    except ValueError:
        pass
    raise ValueError(f"{setting} must be a positive integer number of seconds.")


class LogClient(Protocol):
    """Log access required by the HTTP layer, independent of the training backend."""

    def get_job_logs(self, job_id: str) -> str: ...

    def get_job_status(self, job_id: str) -> Any: ...

    def tail_job_logs(self, job_id: str) -> AsyncIterator[str]: ...


def logs_unavailable() -> ApiError:
    return ApiError(503, "logs_unavailable", "The authenticated job log backend is unavailable.")


def logs_not_found(project: str | None = None, name: str | None = None) -> ApiError:
    target = f" for job '{name}' in project '{project}'" if project and name else ""
    return ApiError(
        404,
        "not_found",
        f"Training submission{target} was not found; driver logs are unavailable.",
    )


def _check_log_auth(response: Any, **kwargs: Any) -> None:
    """Preserve HTTP auth status before Ray normalizes it to AuthenticationError."""
    if response.status_code in {401, 403}:
        raise ApiError(
            response.status_code,
            "unauthorized" if response.status_code == 401 else "forbidden",
            "Ray gateway access was denied.",
            headers={"WWW-Authenticate": "Bearer"} if response.status_code == 401 else None,
        )


async def _reject_log_redirect(
    session: aiohttp.ClientSession, context: Any, parameters: aiohttp.TraceRequestRedirectParams
) -> None:
    """Abort before aiohttp can send caller credentials to a redirected path."""
    parameters.response.close()
    raise logs_unavailable()


class BoundedJobClient(JobSubmissionClient):
    """Bound SDK HTTP calls and never forward caller credentials on redirects."""

    def _do_request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        try:
            response = super()._do_request(
                method,
                endpoint,
                timeout=(5, 15),
                allow_redirects=False,
                hooks={"response": _check_log_auth},
                **kwargs,
            )
        except ApiError:
            raise
        except AuthenticationError:
            raise ApiError(403, "forbidden", "Ray gateway access was denied.") from None
        except Exception:
            # SDK errors may contain raw response bodies, URLs or credentials.
            raise logs_unavailable() from None
        # A missing version or gateway endpoint is not a job readiness failure.
        if (
            response.status_code == 404
            and method == "GET"
            and re.fullmatch(r"/api/jobs/[^/]+(?:/logs)?", endpoint)
        ):
            raise logs_not_found()
        if response.status_code >= 300:
            raise logs_unavailable()
        return response

    async def tail_job_logs(self, job_id: str) -> AsyncIterator[str]:
        """Harden Ray 2.58's WebSocket transport, which bypasses _do_request."""
        # ws_connect has no allow_redirects argument. The public trace callback
        # aborts a redirect before a second request, including on the same origin.
        trace = aiohttp.TraceConfig()
        trace.on_request_redirect.append(_reject_log_redirect)
        async with aiohttp.ClientSession(
            cookies=self._cookies,
            headers=self._headers,
            trace_configs=[trace],
            timeout=aiohttp.ClientTimeout(total=None, connect=15, sock_connect=5),
        ) as session:
            async with session.ws_connect(
                f"{self._address}/api/jobs/{job_id}/logs/tail",
                ssl=self._ssl_context,
            ) as socket:
                async for message in socket:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        yield message.data
                    elif message.type == aiohttp.WSMsgType.ERROR:
                        raise logs_unavailable()
                if socket.close_code == aiohttp.WSCloseCode.ABNORMAL_CLOSURE:
                    raise logs_unavailable()


class CodeFlareLogClient(RayJobClient):
    """Use CodeFlare's log API with a bounded, redirect-safe Ray transport."""

    def __init__(self, address: str, *, headers: dict[str, str], verify: str | bool = True) -> None:
        # CodeFlare SDK 0.40 has no transport injection parameter. Initialize its delegate
        # directly so construction cannot issue an unhardened request first.
        self.rayJobClient = BoundedJobClient(address, headers=headers, verify=verify)


def _snapshot(sdk: LogClient, submission_id: str) -> str:
    try:
        return str(sdk.get_job_logs(submission_id))
    except ApiError:
        raise
    except Exception:
        raise logs_unavailable() from None


def _check_status(sdk: LogClient, submission_id: str) -> None:
    try:
        sdk.get_job_status(submission_id)
    except ApiError:
        raise
    except Exception:
        raise logs_unavailable() from None


async def driver_logs(
    request: Request, project: str, name: str, follow: bool | None, tail_lines: int | None
) -> PlainTextResponse | StreamingResponse:
    backend: JobsBackend = request.app.state.job_backend
    sdk, submission_id = await anyio.to_thread.run_sync(backend.log_client, request, project, name)
    try:
        if not follow:
            text = await anyio.to_thread.run_sync(_snapshot, sdk, submission_id)
            return PlainTextResponse(
                "".join(text.splitlines(keepends=True)[-(tail_lines or 100) :]),
                headers={"Cache-Control": "no-store"},
            )

        # Check the backend without downloading the entire log before streaming it.
        # WebSocket-specific failures are sanitized on the SSE error path below.
        await anyio.to_thread.run_sync(_check_status, sdk, submission_id)
    except ApiError as error:
        if error.status_code == 404 and error.code == "not_found":
            raise logs_not_found(project, name) from None
        raise

    async def events() -> AsyncIterator[str]:
        try:
            # Bound credential lifetime on a stream; reconnect reauthenticates.
            stream = cast(AsyncGenerator[str, None], sdk.tail_job_logs(submission_id))
            async with (
                asyncio.timeout(request.app.state.log_stream_timeout_seconds),
                aclosing(stream) as chunks,
            ):
                async for chunk in chunks:
                    yield "event: log\ndata: " + json.dumps({"message": chunk}) + "\n\n"
        except TimeoutError:
            yield (
                'event: error\ndata: {"code":"stream_expired",'
                '"message":"Reconnect to continue."}\n\n'
            )
        except Exception:
            yield (
                'event: error\ndata: {"code":"logs_unavailable",'
                '"message":"Job log stream failed."}\n\n'
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
