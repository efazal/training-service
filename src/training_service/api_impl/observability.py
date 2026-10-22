"""Ray driver log snapshots and transient SSE streams."""

from fastapi import Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from training_service.ray_logs import driver_logs
from training_service_api.apis.observability_api_base import BaseObservabilityApi
from training_service_api.models.extra_models import TokenModel


class RayObservabilityApi(BaseObservabilityApi):
    async def stream_job_logs(
        self,
        project: str,
        job_id: str,
        follow: bool | None,
        tail_lines: int | None,
        token_OpenShiftBearer: TokenModel,
        request: Request,
    ) -> PlainTextResponse | StreamingResponse:
        return await driver_logs(request, project, job_id, follow, tail_lines)
