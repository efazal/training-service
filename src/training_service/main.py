"""FastAPI application entry point."""

import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from training_service import catalog
from training_service.api_impl import (  # noqa: F401
    infrastructure_discovery,
    observability,
    training_jobs,
)
from training_service.auth import KubernetesTokenAuthenticator, OpenShiftTokenAuthenticator
from training_service.authorization import KubernetesProjectAuthorizer, OpenShiftProjectAuthorizer
from training_service.contract import component_validator, contract_document, load_contract
from training_service.errors import ApiError
from training_service.jobs import CodeFlareJobBackend, JobsBackend, JobSubmitter
from training_service.logs import log_stream_timeout_seconds
from training_service_api.apis.estimations_api import router as estimations_router
from training_service_api.apis.infrastructure_discovery_api import router as discovery_router
from training_service_api.apis.observability_api import router as observability_router
from training_service_api.apis.training_jobs_api import router as jobs_router
from training_service_api.models.error_response import ErrorResponse

_SUBMISSION_PATH = re.compile(r"^/api/v1/projects/[^/]+/training-jobs$")
_LOG = logging.getLogger(__name__)
_SAFE_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _submission_validation_message(body: object) -> str:
    """Explain contract failures using field names, never user-supplied values."""
    if not isinstance(body, dict):
        return "Request body must be a JSON object."
    issue = next(component_validator("CreateJobRequest").iter_errors(body), None)
    if issue is None:
        return "Request does not match the API contract."
    path = [str(part) for part in issue.path]
    if any(_SAFE_FIELD.fullmatch(part) is None for part in path):
        return "Request does not match the API contract."
    field = ".".join(path)
    if issue.validator == "required" and isinstance(issue.instance, dict):
        missing = next((key for key in issue.validator_value if key not in issue.instance), None)
        if isinstance(missing, str) and _SAFE_FIELD.fullmatch(missing):
            return f"Missing required field: {field + '.' if field else ''}{missing}."
    if issue.validator == "additionalProperties" and isinstance(issue.instance, dict):
        known = issue.schema.get("properties", {})
        extra = next((key for key in issue.instance if key not in known), None)
        if isinstance(extra, str) and _SAFE_FIELD.fullmatch(extra):
            return f"Unsupported field: {field + '.' if field else ''}{extra}."
    if field:
        if issue.validator == "maxLength":
            return f"{field} must be at most {issue.validator_value} characters."
        if issue.validator == "minLength":
            return f"{field} must be at least {issue.validator_value} character(s)."
        if issue.validator == "type":
            expected = issue.validator_value
            if isinstance(expected, list):
                return f"{field} must be one of these types: {', '.join(expected)}."
            article = "an" if expected in {"integer", "object", "array"} else "a"
            return f"{field} must be {article} {expected}."
        if issue.validator in {"minimum", "maximum"}:
            direction = "greater" if issue.validator == "minimum" else "less"
            exclusive = issue.schema.get("exclusive" + issue.validator.title(), False)
            comparator = " than" if exclusive else " than or equal to"
            return f"{field} must be {direction}{comparator} {issue.validator_value}."
        if issue.validator == "pattern":
            return f"{field} has an invalid format."
        if issue.validator == "enum":
            return f"{field} has an unsupported value."
        return f"Invalid value for {field}."
    return "Request does not match the API contract."


def create_app(
    authenticator: OpenShiftTokenAuthenticator | None = None,
    authorizer: OpenShiftProjectAuthorizer | None = None,
    submitter: JobSubmitter | None = None,
    backend: JobsBackend | None = None,
) -> FastAPI:
    """Create the HTTP application."""
    contract = load_contract()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            await anyio.to_thread.run_sync(catalog.load_catalog)
        except ApiError as error:
            # Keep diagnostics alive, but never report readiness with a bad catalog.
            _LOG.error("%s", error.message)
        yield

    application = FastAPI(
        title=contract["info"]["title"],
        version=contract["info"]["version"],
        description=contract["info"]["description"],
        swagger_ui_parameters={"persistAuthorization": False},
        lifespan=lifespan,
    )

    application.state.log_stream_timeout_seconds = log_stream_timeout_seconds()
    application.state.token_authenticator = authenticator or KubernetesTokenAuthenticator()
    application.state.project_authorizer = authorizer or KubernetesProjectAuthorizer()
    application.state.job_backend = backend or CodeFlareJobBackend()
    application.state.job_submitter = submitter or application.state.job_backend

    @application.middleware("http")
    async def require_json_submission(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "POST" and _SUBMISSION_PATH.fullmatch(request.url.path):
            media_type = request.headers.get("content-type", "").partition(";")[0].lower()
            if media_type != "application/json":
                return JSONResponse(
                    status_code=415,
                    content={
                        "code": "unsupported_media_type",
                        "message": "Request body must use application/json.",
                    },
                )
        return await call_next(request)

    @application.exception_handler(ApiError)
    async def api_error_handler(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorResponse(code=error.code, message=error.message).model_dump(
                exclude_unset=True
            ),
            headers=error.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        # Query errors remain 400; submission preserves its 400/422 envelopes.
        if request.method in {"GET", "DELETE"}:
            return JSONResponse(
                status_code=400,
                content={"code": "bad_request", "message": "Invalid path or query parameters."},
            )
        malformed_json = any(item.get("type") == "json_invalid" for item in error.errors())
        return JSONResponse(
            status_code=400 if malformed_json else 422,
            content={
                "code": "bad_request" if malformed_json else "validation_error",
                "message": (
                    "Malformed JSON request body."
                    if malformed_json
                    else _submission_validation_message(error.body)
                    if _SUBMISSION_PATH.fullmatch(request.url.path)
                    else "Request does not match the API contract."
                ),
            },
        )

    for router in (estimations_router, discovery_router, observability_router, jobs_router):
        application.include_router(router, prefix="/api/v1")
    application.openapi = contract_document  # type: ignore[method-assign]

    @application.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz", include_in_schema=False)
    def ready() -> dict[str, str]:
        catalog.load_catalog()
        return {"status": "ok"}

    return application


app = create_app()


def run() -> None:
    """Run the service locally."""
    import uvicorn

    uvicorn.run("training_service.main:app", host="0.0.0.0", port=8080)
