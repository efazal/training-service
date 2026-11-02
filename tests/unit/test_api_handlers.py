"""Single handler serves all training job operations."""

import pytest

from training_service.api_impl.training_jobs import TrainingJobsApi
from training_service.contract import load_contract
from training_service.errors import ApiError
from training_service_api.apis.training_jobs_api_base import BaseTrainingJobsApi
from training_service_api.models.error_response import ErrorResponse


def test_single_handler_serves_all_operations():
    implementation = BaseTrainingJobsApi.implementation()
    assert isinstance(implementation, TrainingJobsApi)
    for operation in (
        "create_training_job",
        "list_training_jobs",
        "get_training_job",
        "delete_training_job",
    ):
        assert operation in TrainingJobsApi.__dict__


def test_no_handler_fails_closed(monkeypatch):
    monkeypatch.setattr(BaseTrainingJobsApi, "subclasses", ())
    with pytest.raises(ApiError) as raised:
        BaseTrainingJobsApi.implementation()
    assert raised.value.status_code == 501
    assert raised.value.code == "not_implemented"


def test_duplicate_handlers_are_rejected(monkeypatch):
    monkeypatch.setattr(BaseTrainingJobsApi, "subclasses", (TrainingJobsApi, TrainingJobsApi))
    with pytest.raises(RuntimeError, match="Expected one implementation"):
        BaseTrainingJobsApi.implementation()


@pytest.mark.parametrize(
    "code",
    [
        "submission_unavailable",
        "backend_unavailable",
        "conflict",
        "invalid_page_token",
        "unowned_cluster",
        "shared_cluster",
        "logs_unavailable",
        "stream_expired",
    ],
)
def test_error_codes_match_the_contract(code):
    assert (
        code
        in load_contract()["components"]["schemas"]["ErrorResponse"]["properties"]["code"]["enum"]
    )
    assert ErrorResponse(code=code, message="Sanitized error.").code == code


def test_handler_is_registered_via_base_subclasses():
    assert TrainingJobsApi in BaseTrainingJobsApi.subclasses
