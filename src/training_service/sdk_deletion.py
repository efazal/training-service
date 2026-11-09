"""Reattach the pinned CodeFlare SDK to a checked, caller-scoped RayJob."""

from typing import Any

from codeflare_sdk import RayJob
from codeflare_sdk.vendored.python_client.kuberay_job_api import RayjobApi
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from training_service.kubernetes_client import TIMEOUT


class _DeleteError(Exception):
    """Carry Kubernetes errors through CodeFlare 0.40's catch-and-return wrapper."""

    def __init__(self, error: ApiException) -> None:
        super().__init__("The SDK RayJob deletion failed.")
        self.error = error


class _DeleteTransport:
    """Add the preconditions and cascading policy missing from the SDK delete API."""

    def __init__(self, api: client.ApiClient, metadata: dict[str, Any]) -> None:
        self.api = client.CustomObjectsApi(api)
        self.options = client.V1DeleteOptions(
            propagation_policy="Foreground",
            preconditions=client.V1Preconditions(
                uid=metadata["uid"], resource_version=metadata["resourceVersion"]
            ),
        )

    def delete_namespaced_custom_object(self, **kwargs: Any) -> Any:
        try:
            return self.api.delete_namespaced_custom_object(
                **kwargs, body=self.options, _request_timeout=TIMEOUT
            )
        except ApiException as error:
            # RayjobApi catches ApiException and RayJob.delete then reports success,
            # including for 403/409/500. Preserve those statuses for the HTTP layer.
            raise _DeleteError(error) from None


def delete_ray_job(api: client.ApiClient, project: str, job: dict[str, Any]) -> None:
    """Use RayJob.delete without loading process-global or service-account credentials."""
    # SDK 0.40 has no attach-existing constructor. Its normal constructors load
    # global kubeconfig, create submission state, and cannot accept delete options.
    # Hydrate only the fields used by the real, pinned RayJob.delete implementation.
    sdk = object.__new__(RayJob)
    sdk.name = job["metadata"]["name"]
    sdk.namespace = project
    sdk._api_client = api
    sdk._api = object.__new__(RayjobApi)
    sdk._api.api = _DeleteTransport(api, job["metadata"])
    try:
        sdk.delete()
    except _DeleteError as error:
        raise error.error from None
