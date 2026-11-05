"""Backend-neutral infrastructure discovery handlers."""

import anyio
from fastapi import Request

from training_service import algorithm_discovery
from training_service_api.apis.infrastructure_discovery_api_base import (
    BaseInfrastructureDiscoveryApi,
)
from training_service_api.models.algorithm_details import AlgorithmDetails
from training_service_api.models.extra_models import TokenModel
from training_service_api.models.list_algorithms200_response import ListAlgorithms200Response


class InfrastructureDiscoveryApi(BaseInfrastructureDiscoveryApi):
    async def list_algorithms(
        self, token_OpenShiftBearer: TokenModel, request: Request
    ) -> ListAlgorithms200Response:
        result = await anyio.to_thread.run_sync(algorithm_discovery.list_algorithms, request)
        return ListAlgorithms200Response.model_validate(result)

    async def get_algorithm(
        self, name: str, include: str | None, token_OpenShiftBearer: TokenModel, request: Request
    ) -> AlgorithmDetails:
        result = await anyio.to_thread.run_sync(
            algorithm_discovery.get_algorithm, request, name, include
        )
        return AlgorithmDetails.model_validate(result)
