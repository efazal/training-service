"""Authenticated Ray SDK driver logs through an administrator-trusted gateway."""

import asyncio
import json
import os
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any, cast
from urllib.parse import urlsplit

import aiohttp
import anyio
from fastapi import Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from kubernetes.client.exceptions import ApiException
from ray.exceptions import AuthenticationError
from ray.job_submission import JobSubmissionClient

from training_service.errors import ApiError
from training_service.jobs import owned_cluster, read_job
from training_service.kube import caller_client, validate_names


def _unavailable() -> ApiError:
    return ApiError(503, "logs_unavailable", "The authenticated Ray log backend is unavailable.")


async def _reject_log_redirect(
    session: aiohttp.ClientSession, context: Any, parameters: aiohttp.TraceRequestRedirectParams
) -> None:
    """Abort before aiohttp can send caller credentials to a redirected path."""
    parameters.response.close()
    raise _unavailable()


class BoundedJobClient(JobSubmissionClient):
    """Bound SDK HTTP calls and never forward caller credentials on redirects."""

    def _do_request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        try:
            response = super()._do_request(
                method, endpoint, timeout=(5, 15), allow_redirects=False, **kwargs
            )
        except AuthenticationError:
            raise ApiError(403, "forbidden", "Ray gateway access was denied.") from None
        except Exception:
            # SDK errors may contain raw response bodies, URLs or credentials.
            raise _unavailable() from None
        if response.status_code in {401, 403}:
            raise ApiError(response.status_code, "forbidden", "Ray gateway access was denied.")
        if response.status_code == 404:
            raise ApiError(409, "logs_not_ready", "Driver logs are not available yet; retry later.")
        if response.status_code >= 300:
            raise _unavailable()
        return response

    async def tail_job_logs(self, job_id: str) -> AsyncIterator[str]:
        """Harden Ray 2.55's WebSocket transport, which bypasses _do_request."""
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
                        raise _unavailable()
                if socket.close_code == aiohttp.WSCloseCode.ABNORMAL_CLOSURE:
                    raise _unavailable()


def resolve_client(request: Request, project: str, name: str) -> tuple[JobSubmissionClient, str]:
    validate_names(project, name)
    with caller_client(request) as api:
        job = read_job(api, project, name)
        submission_id = job.get("status", {}).get("jobId")
        try:
            cluster = owned_cluster(api, project, job)
        except ApiException as error:
            if error.status == 404:
                raise ApiError(
                    410,
                    "ray_cluster_gone",
                    "Driver logs are gone because the RayCluster was removed.",
                ) from None
            raise
        except ApiError as error:
            if error.code == "unowned_cluster":
                raise ApiError(
                    403, "forbidden", "RayCluster ownership could not be verified."
                ) from None
            raise
    if not cluster or not submission_id:
        raise ApiError(409, "logs_not_ready", "Driver logs are not available yet; retry later.")
    if (
        not isinstance(submission_id, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,1024}", submission_id) is None
    ):
        raise _unavailable()
    gateway = os.getenv("TRAINING_SERVICE_RAY_GATEWAY_URL", "").rstrip("/")
    try:
        parsed = urlsplit(gateway)
    except ValueError:
        raise _unavailable() from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or os.getenv("RAY_ADDRESS")
        or os.getenv("RAY_API_SERVER_ADDRESS")
    ):
        raise _unavailable()
    try:
        sdk = BoundedJobClient(
            f"{gateway}/ray/{project}/{cluster}",
            headers={"Authorization": request.headers["authorization"]},
            verify=os.getenv("TRAINING_SERVICE_RAY_CA_FILE") or True,
        )
    except ApiError:
        raise
    except Exception:
        raise _unavailable() from None
    return sdk, submission_id


def _snapshot(sdk: JobSubmissionClient, submission_id: str) -> str:
    try:
        return str(sdk.get_job_logs(submission_id))
    except ApiError:
        raise
    except Exception:
        raise _unavailable() from None


def _check_status(sdk: JobSubmissionClient, submission_id: str) -> None:
    try:
        sdk.get_job_status(submission_id)
    except ApiError:
        raise
    except Exception:
        raise _unavailable() from None


async def driver_logs(
    request: Request, project: str, name: str, follow: bool | None, tail_lines: int | None
) -> PlainTextResponse | StreamingResponse:
    sdk, submission_id = await anyio.to_thread.run_sync(resolve_client, request, project, name)
    if not follow:
        text = await anyio.to_thread.run_sync(_snapshot, sdk, submission_id)
        return PlainTextResponse(
            "".join(text.splitlines(keepends=True)[-(tail_lines or 100) :]),
            headers={"Cache-Control": "no-store"},
        )

    # Check the backend without downloading the entire log before streaming it.
    # WebSocket-specific failures are sanitized on the SSE error path below.
    await anyio.to_thread.run_sync(_check_status, sdk, submission_id)

    async def events() -> AsyncIterator[str]:
        try:
            # Bound credential lifetime on a stream; reconnect reauthenticates.
            stream = cast(AsyncGenerator[str, None], sdk.tail_job_logs(submission_id))
            async with asyncio.timeout(300), aclosing(stream) as chunks:
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
                '"message":"Ray log stream failed."}\n\n'
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
