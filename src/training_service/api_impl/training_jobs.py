"""Handwritten Training Jobs implementation; only submission is in this ticket."""

from __future__ import annotations

from typing import Any, NoReturn

from fastapi import Request

from training_service.auth import CallerIdentity
from training_service.errors import ApiError
from training_service.job_submission import JobSubmitter
from training_service_api.apis.training_jobs_api_base import BaseTrainingJobsApi
from training_service_api.models.create_job_request import CreateJobRequest
from training_service_api.models.extra_models import TokenModel
from training_service_api.models.training_job_submission import TrainingJobSubmission


def _not_implemented() -> NoReturn:
    raise ApiError(501, "not_implemented", "This operation is not implemented.")


class SubmissionTrainingJobsApi(BaseTrainingJobsApi):
    """Register one generated handler while other operations remain fail-closed."""

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
            "selected_backend": "ray",
        }
        if create_job_request.queue_name is not None:
            receipt["queue_name"] = create_job_request.queue_name
        return TrainingJobSubmission(
            **receipt,
        )

    async def list_training_jobs(self, *args: Any, **kwargs: Any) -> NoReturn:
        _not_implemented()

    async def get_training_job(self, *args: Any, **kwargs: Any) -> NoReturn:
        _not_implemented()

    async def delete_training_job(self, *args: Any, **kwargs: Any) -> NoReturn:
        _not_implemented()
