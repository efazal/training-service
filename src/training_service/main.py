"""FastAPI application entry point."""

import re
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from training_service.api_impl import job_operations, observability, training_jobs  # noqa: F401
from training_service.auth import KubernetesTokenAuthenticator, OpenShiftTokenAuthenticator
from training_service.authorization import KubernetesProjectAuthorizer, OpenShiftProjectAuthorizer
from training_service.contract import contract_document, load_contract
from training_service.errors import ApiError
from training_service.job_submission import CodeFlareJobSubmitter, JobSubmitter
from training_service_api.apis.estimations_api import router as estimations_router
from training_service_api.apis.infrastructure_discovery_api import router as discovery_router
from training_service_api.apis.observability_api import router as observability_router
from training_service_api.apis.training_jobs_api import router as jobs_router
from training_service_api.models.error_response import ErrorResponse

_SUBMISSION_PATH = re.compile(r"^/api/v1/projects/[^/]+/training-jobs$")


def create_app(
    authenticator: OpenShiftTokenAuthenticator | None = None,
    authorizer: OpenShiftProjectAuthorizer | None = None,
    submitter: JobSubmitter | None = None,
) -> FastAPI:
    """Create the HTTP application."""
    contract = load_contract()
    application = FastAPI(
        title=contract["info"]["title"],
        version=contract["info"]["version"],
        description=contract["info"]["description"],
        swagger_ui_parameters={"persistAuthorization": False},
    )

    application.state.token_authenticator = authenticator or KubernetesTokenAuthenticator()
    application.state.project_authorizer = authorizer or KubernetesProjectAuthorizer()
    application.state.job_submitter = submitter or CodeFlareJobSubmitter()

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
        return {"status": "ok"}

    return application


app = create_app()


def run() -> None:
    """Run the service locally."""
    import uvicorn

    uvicorn.run("training_service.main:app", host="0.0.0.0", port=8080)
