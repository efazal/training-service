"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from training_service.api_impl import observability, training_jobs  # noqa: F401
from training_service.auth import KubernetesTokenAuthenticator, OpenShiftTokenAuthenticator
from training_service.authorization import KubernetesProjectAuthorizer, OpenShiftProjectAuthorizer
from training_service.contract import contract_document, load_contract
from training_service.errors import ApiError
from training_service_api.apis.estimations_api import router as estimations_router
from training_service_api.apis.infrastructure_discovery_api import router as discovery_router
from training_service_api.apis.observability_api import router as observability_router
from training_service_api.apis.training_jobs_api import router as jobs_router


def create_app(
    authenticator: OpenShiftTokenAuthenticator | None = None,
    authorizer: OpenShiftProjectAuthorizer | None = None,
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

    @application.exception_handler(ApiError)
    async def api_error_handler(request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"code": error.code, "message": error.message},
            headers=error.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        # Query/path errors use the reviewed 400 envelope; do not reflect input.
        if request.method in {"GET", "DELETE"}:
            return JSONResponse(
                status_code=400,
                content={"code": "bad_request", "message": "Invalid path or query parameters."},
            )
        from fastapi.exception_handlers import request_validation_exception_handler

        return await request_validation_exception_handler(request, error)

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
