"""Driver log snapshots and transient SSE streams."""

import re

from fastapi import Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from training_service.errors import ApiError
from training_service.logs import driver_logs
from training_service_api.apis.observability_api_base import BaseObservabilityApi
from training_service_api.models.extra_models import TokenModel


class ObservabilityApi(BaseObservabilityApi):
    async def stream_job_logs(
        self,
        project: str,
        job_id: str,
        follow: bool | None,
        tail_lines: int | None,
        timeout_seconds: int | None,
        token_OpenShiftBearer: TokenModel,
        request: Request,
    ) -> PlainTextResponse | StreamingResponse:
        # Query values arrive as strings; strict=True would reject valid URL integers.
        if (
            timeout_seconds is not None
            and re.fullmatch(r"[1-9][0-9]*", request.query_params.get("timeout_seconds", ""))
            is None
        ):
            raise ApiError(
                400,
                "bad_request",
                "timeout_seconds must be a positive integer written with ASCII digits "
                "and no leading zeros.",
            )
        return await driver_logs(request, project, job_id, follow, tail_lines, timeout_seconds)
