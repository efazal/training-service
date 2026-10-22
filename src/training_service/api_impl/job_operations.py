"""Handwritten job reads and deletion outside the generated API package."""

from typing import ClassVar

import anyio
from fastapi import Request, Response

from training_service import jobs
from training_service_api.apis.training_jobs_api_base import BaseTrainingJobsApi
from training_service_api.models.extra_models import TokenModel
from training_service_api.models.job_status import JobStatus
from training_service_api.models.list_training_jobs200_response import ListTrainingJobs200Response
from training_service_api.models.training_job import TrainingJob


class RayTrainingJobsApi(BaseTrainingJobsApi):
    """Generated routes perform authentication and project SAR before dispatch."""

    implemented_operations: ClassVar[frozenset[str]] = frozenset(
        {
            "list_training_jobs",
            "get_training_job",
            "delete_training_job",
        }
    )

    async def list_training_jobs(
        self,
        project: str,
        status: JobStatus | None,
        queue_name: str | None,
        limit: int | None,
        page_token: str | None,
        token_OpenShiftBearer: TokenModel,
        request: Request,
    ) -> ListTrainingJobs200Response:
        result = await anyio.to_thread.run_sync(
            jobs.list_jobs,
            request,
            project,
            status.value if status else None,
            queue_name,
            limit,
            page_token,
        )
        return ListTrainingJobs200Response.model_validate(result)

    async def get_training_job(
        self,
        project: str,
        job_id: str,
        token_OpenShiftBearer: TokenModel,
        request: Request,
    ) -> TrainingJob:
        result = await anyio.to_thread.run_sync(jobs.get_job, request, project, job_id)
        return TrainingJob.model_validate(result)

    async def delete_training_job(
        self,
        project: str,
        job_id: str,
        token_OpenShiftBearer: TokenModel,
        request: Request,
    ) -> Response:
        await anyio.to_thread.run_sync(jobs.delete_job, request, project, job_id)
        return Response(status_code=204)
