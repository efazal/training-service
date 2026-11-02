"""Training Jobs API handler: submission, reads, and deletion."""

from __future__ import annotations

import anyio
from fastapi import Request, Response

from training_service.auth import CallerIdentity
from training_service.errors import ApiError
from training_service.jobs import JobsBackend, JobSubmitter
from training_service_api.apis.training_jobs_api_base import BaseTrainingJobsApi
from training_service_api.models.create_job_request import CreateJobRequest
from training_service_api.models.extra_models import TokenModel
from training_service_api.models.job_status import JobStatus
from training_service_api.models.list_training_jobs200_response import ListTrainingJobs200Response
from training_service_api.models.training_job import TrainingJob
from training_service_api.models.training_job_submission import TrainingJobSubmission


class TrainingJobsApi(BaseTrainingJobsApi):
    async def create_training_job(
        self,
        project: str,
        create_job_request: CreateJobRequest,
        token_OpenShiftBearer: TokenModel,
        request: Request,
    ) -> TrainingJobSubmission:
        del token_OpenShiftBearer
        caller = getattr(request.state, "caller_identity", None)
        token = getattr(request.state, "caller_bearer_token", None)
        if not isinstance(caller, CallerIdentity) or not isinstance(token, str):
            raise ApiError(403, "forbidden", "Verified caller context is unavailable.")
        submitter: JobSubmitter = request.app.state.job_submitter
        identifier = await submitter.submit(project, create_job_request, caller, token)
        receipt: dict[str, str] = {
            "id": identifier,
            "name": create_job_request.name,
            "project": project,
            "status": "SUBMITTED",
            "selected_backend": request.app.state.job_backend.backend_name,
        }
        if create_job_request.queue_name is not None:
            receipt["queue_name"] = create_job_request.queue_name
        return TrainingJobSubmission(**receipt)

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
        backend: JobsBackend = request.app.state.job_backend
        result = await anyio.to_thread.run_sync(
            backend.list_jobs,
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
        backend: JobsBackend = request.app.state.job_backend
        result = await anyio.to_thread.run_sync(backend.get_job, request, project, job_id)
        return TrainingJob.model_validate(result)

    async def delete_training_job(
        self,
        project: str,
        job_id: str,
        token_OpenShiftBearer: TokenModel,
        request: Request,
    ) -> Response:
        backend: JobsBackend = request.app.state.job_backend
        await anyio.to_thread.run_sync(backend.delete_job, request, project, job_id)
        return Response(status_code=204)
