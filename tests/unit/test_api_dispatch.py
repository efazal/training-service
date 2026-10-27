"""Submission and read handlers must coexist independently of import order."""

from types import SimpleNamespace

import pytest

from training_service.api_impl.dispatch import implementation_for
from training_service.api_impl.job_operations import RayTrainingJobsApi
from training_service.api_impl.training_jobs import SubmissionTrainingJobsApi
from training_service.contract import load_contract
from training_service.errors import ApiError
from training_service_api.models.error_response import ErrorResponse


@pytest.mark.parametrize("submission_first", [False, True])
def test_explicit_read_owner_wins_over_submission_stubs(submission_first):
    handlers = (SubmissionTrainingJobsApi, RayTrainingJobsApi)
    base = SimpleNamespace(subclasses=handlers if submission_first else handlers[::-1])
    assert isinstance(implementation_for(base, "create_training_job"), SubmissionTrainingJobsApi)
    for operation in RayTrainingJobsApi.implemented_operations:
        assert isinstance(implementation_for(base, operation), RayTrainingJobsApi)


def test_missing_submission_fails_closed():
    with pytest.raises(ApiError) as raised:
        implementation_for(SimpleNamespace(subclasses=(RayTrainingJobsApi,)), "create_training_job")
    assert raised.value.status_code == 501
    assert raised.value.code == "not_implemented"


def test_ambiguous_owners_are_not_selected_by_import_order():
    class DuplicateReads:
        implemented_operations = RayTrainingJobsApi.implemented_operations

    with pytest.raises(RuntimeError, match="Multiple implementations"):
        implementation_for(
            SimpleNamespace(subclasses=(RayTrainingJobsApi, DuplicateReads)), "get_training_job"
        )
    with pytest.raises(RuntimeError, match="Multiple legacy implementations"):
        implementation_for(
            SimpleNamespace(subclasses=(SubmissionTrainingJobsApi, SubmissionTrainingJobsApi)),
            "create_training_job",
        )


@pytest.mark.parametrize(
    "code",
    [
        "invalid_request",
        "submission_unavailable",
        "backend_not_implemented",
        "backend_unavailable",
        "conflict",
        "invalid_page_token",
        "unowned_cluster",
        "shared_cluster",
        "logs_unavailable",
        "stream_expired",
    ],
)
def test_submission_and_read_errors_match_the_contract(code):
    assert (
        code
        in load_contract()["components"]["schemas"]["ErrorResponse"]["properties"]["code"]["enum"]
    )
    assert ErrorResponse(code=code, message="Sanitized error.").code == code
