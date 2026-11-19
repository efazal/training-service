"""Fail-closed Kueue capability checks shared by discovery and queued submission."""

import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import yaml
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from urllib3.exceptions import HTTPError

from training_service.errors import ApiError
from training_service.kubernetes_client import TIMEOUT, describe_not_found

GROUP = "kueue.x-k8s.io"


def unavailable() -> ApiError:
    return ApiError(
        503,
        "backend_unavailable",
        "Kueue readiness could not be established. "
        "Contact the platform administrator or retry later.",
    )


@contextmanager
def platform_client() -> Iterator[client.ApiClient]:
    """Never substitute this identity for caller-owned project or queue reads."""
    try:
        configuration = client.Configuration()
        config.load_incluster_config(client_configuration=configuration)
        configuration.retries = 0
        with client.ApiClient(configuration) as api:
            yield api
    except (ApiException, ConfigException, HTTPError, OSError) as error:
        # A service-account denial is a deployment problem, not a caller denial.
        raise unavailable() from error


def matches(selector: Any, labels: Mapping[str, str]) -> bool:
    """Evaluate Kubernetes LabelSelector semantics; reject unknown configuration."""
    if not isinstance(selector, dict) or set(selector) - {"matchLabels", "matchExpressions"}:
        raise unavailable()
    pairs = selector.get("matchLabels", {})
    expressions = selector.get("matchExpressions", [])
    pairs = {} if pairs is None else pairs
    expressions = [] if expressions is None else expressions
    if not isinstance(pairs, dict) or not isinstance(expressions, list):
        raise unavailable()
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in pairs.items()):
        raise unavailable()
    result = all(labels.get(k) == v for k, v in pairs.items())
    for expression in expressions:
        if not isinstance(expression, dict):
            raise unavailable()
        key, op, values = (
            expression.get("key"),
            expression.get("operator"),
            expression.get("values", []),
        )
        if (
            not isinstance(key, str)
            or not isinstance(op, str)
            or not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
        ):
            raise unavailable()
        if op in {"In", "NotIn"} and values:
            match = labels.get(key) in values if op == "In" else labels.get(key) not in values
        elif op in {"Exists", "DoesNotExist"} and not values:
            match = key in labels if op == "Exists" else key not in labels
        else:
            raise unavailable()
        result = result and match
    return result


def served_version(api: client.ApiClient) -> str:
    """Discover preferred served LocalQueue version, not a fixed beta-version chain."""
    groups = client.ApisApi(api).get_api_versions(_request_timeout=TIMEOUT)
    group = next((g for g in groups.groups if g.name == GROUP), None)
    if group is None:
        raise ApiError(409, "kueue_not_installed", "Kueue is not installed on this cluster.")
    versions = [v.version for v in group.versions]
    if group.preferred_version and group.preferred_version.version in versions:
        preferred = group.preferred_version.version
        versions = [preferred] + [v for v in versions if v != preferred]
    for version in versions:
        if serves_localqueues(api, version):
            return str(version)
    # A retained/incompatible API is not proof of an absent installation.
    raise unavailable()


def serves_localqueues(api: client.ApiClient, version: str) -> bool:
    """Validate the discovery document before trusting resource capabilities."""
    resources = api.call_api(
        f"/apis/{GROUP}/{version}",
        "GET",
        response_type="object",
        auth_settings=["BearerToken"],
        _return_http_data_only=True,
        _request_timeout=TIMEOUT,
    )
    if not isinstance(resources, dict) or not isinstance(resources.get("resources"), list):
        raise unavailable()
    found = False
    for resource in resources["resources"]:
        if not isinstance(resource, dict):
            raise unavailable()
        verbs = resource.get("verbs")
        if not isinstance(verbs, list) or not all(isinstance(v, str) for v in verbs):
            raise unavailable()
        if (
            resource.get("name") == "localqueues"
            and resource.get("namespaced") is True
            and {"get", "list"} <= set(verbs)
        ):
            found = True
    return found


def valid_queue_name(name: Any) -> bool:
    """Queue DNS subdomain, bounded by the RayJob queue label's 63-character limit."""
    label = r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?"
    return (
        isinstance(name, str)
        and len(name) <= 63
        and re.fullmatch(rf"{label}(?:\.{label})*", name) is not None
    )


def ready_webhook_port(reference: Any, service: Any, endpoints: Any) -> bool:
    """Match the admission Service port to a ready backend, not any ready address."""
    port_number = reference.port if reference.port is not None else 443
    for port in service.spec.ports or []:
        if port.port != port_number or (port.protocol or "TCP") != "TCP":
            continue
        target = port.target_port if port.target_port is not None else port.port
        for subset in endpoints.subsets or []:
            if not subset.addresses:
                continue
            for endpoint in subset.ports or []:
                if (endpoint.name or "") != (port.name or "") or (
                    endpoint.protocol or "TCP"
                ) != "TCP":
                    continue
                # Named targetPorts are resolved by Kubernetes; EndpointPort.name is
                # the ServicePort name, not the container port's name.
                if isinstance(target, str) or endpoint.port == target:
                    return True
    return False


def capability(labels: Mapping[str, str]) -> str:
    """Read explicitly configured installation resources, independently of DSC."""
    namespace = os.getenv("TRAINING_KUEUE_NAMESPACE", "openshift-kueue-operator")
    deployment = os.getenv("TRAINING_KUEUE_DEPLOYMENT", "kueue-controller-manager")
    configmap = os.getenv("TRAINING_KUEUE_CONFIGMAP", "kueue-manager-config")
    webhook = os.getenv("TRAINING_KUEUE_WEBHOOK", "kueue-mutating-webhook-configuration")
    with platform_client() as api:
        version = served_version(api)
        dep = client.AppsV1Api(api).read_namespaced_deployment(
            deployment, namespace, _request_timeout=TIMEOUT
        )
        if not dep.status or (dep.status.available_replicas or 0) < 1:
            raise unavailable()
        cm = client.CoreV1Api(api).read_namespaced_config_map(
            configmap, namespace, _request_timeout=TIMEOUT
        )
        try:
            settings = yaml.safe_load((cm.data or {})["controller_manager_config.yaml"])
        except (KeyError, yaml.YAMLError) as error:
            raise unavailable() from error
        integrations = settings.get("integrations") if isinstance(settings, dict) else None
        frameworks = integrations.get("frameworks") if isinstance(integrations, dict) else None
        if (
            not isinstance(frameworks, list)
            or not all(isinstance(value, str) for value in frameworks)
            or "ray.io/rayjob" not in frameworks
        ):
            raise unavailable()
        # Require explicit configuration instead of guessing version-dependent defaults.
        selector = settings.get("managedJobsNamespaceSelector")
        eligible = matches(selector, labels)
        configuration = client.AdmissionregistrationV1Api(api).read_mutating_webhook_configuration(
            webhook, _request_timeout=TIMEOUT
        )
        candidates = [
            w
            for w in configuration.webhooks or []
            if w.client_config
            and w.client_config.service
            and w.client_config.service.namespace == namespace
            and w.client_config.service.path == "/mutate-ray-io-v1-rayjob"
        ]
        if len(candidates) != 1:
            raise unavailable()
        admission = candidates[0]
        object_selector = api.sanitize_for_serialization(admission.object_selector) or {}
        if (
            admission.failure_policy != "Fail"
            or admission.match_conditions
            or not isinstance(object_selector, dict)
            or set(object_selector) - {"matchLabels", "matchExpressions"}
            or object_selector.get("matchLabels")
            or object_selector.get("matchExpressions")
        ):
            raise unavailable()
        if not any(
            "ray.io" in (r.api_groups or [])
            and "v1" in (r.api_versions or [])
            and "rayjobs" in (r.resources or [])
            and bool({"CREATE", "*"}.intersection(r.operations or []))
            and r.scope in (None, "Namespaced", "*")
            for r in admission.rules or []
        ):
            raise unavailable()
        service_name = os.getenv("TRAINING_KUEUE_SERVICE", "kueue-webhook-service")
        if admission.client_config.service.name != service_name:
            raise unavailable()
        service = client.CoreV1Api(api).read_namespaced_service(
            service_name, namespace, _request_timeout=TIMEOUT
        )
        if not service.spec.selector or not all(
            dep.spec.template.metadata.labels.get(k) == v for k, v in service.spec.selector.items()
        ):
            raise unavailable()
        endpoints = client.CoreV1Api(api).read_namespaced_endpoints(
            service_name, namespace, _request_timeout=TIMEOUT
        )
        if not ready_webhook_port(admission.client_config.service, service, endpoints):
            raise unavailable()
        webhook_selector = api.sanitize_for_serialization(admission.namespace_selector) or {}
        eligible = matches(webhook_selector, labels) and eligible
        if not eligible:
            raise ApiError(
                409,
                "kueue_project_excluded",
                "The project is excluded by the installed Kueue namespace policy.",
            )
        return version


def preflight(api: client.ApiClient, project: str, queue: str | None = None) -> str:
    """Caller project access, platform capability, then optional caller queue lookup."""
    with describe_not_found("Project", project):
        namespace = client.CoreV1Api(api).read_namespace(project, _request_timeout=TIMEOUT)
    if namespace.metadata is None:
        raise unavailable()
    version = capability(namespace.metadata.labels or {})
    if queue is not None:
        check_queue(api, project, queue, version)
    return version


def check_queue(api: client.ApiClient, project: str, queue: str, version: str) -> None:
    """A missing API version or failed lookup must not be reported as a missing queue."""
    with describe_not_found("LocalQueue", queue, project=project):
        try:
            client.CustomObjectsApi(api).get_namespaced_custom_object(
                GROUP,
                version,
                project,
                "localqueues",
                queue,
                _request_timeout=TIMEOUT,
            )
        except ApiException as error:
            if error.status == 404:
                with platform_client() as platform:
                    if not serves_localqueues(platform, version):
                        raise unavailable() from error
            elif error.status not in (401, 403):
                # Any other status proves neither presence nor absence.
                raise unavailable() from error
            raise


def default_queue(api: client.ApiClient, project: str) -> str | None:
    """Resolve defaults with caller permissions and a served API, never the SDK helper.

    Proven absence or a complete list without a default permits unqueued submission.
    Failed discovery/listing must not silently turn a queued job into an unqueued job.
    """
    with platform_client() as platform:
        try:
            version = served_version(platform)
        except ApiError as error:
            if error.code == "kueue_not_installed":
                return None
            raise
    with describe_not_found("Project", project):
        client.CoreV1Api(api).read_namespace(project, _request_timeout=TIMEOUT)
    try:
        queues = client.CustomObjectsApi(api).list_namespaced_custom_object(
            GROUP,
            version,
            project,
            "localqueues",
            _request_timeout=TIMEOUT,
        )
    except ApiException as error:
        if error.status in (401, 403):
            raise
        raise unavailable() from error
    if not isinstance(queues, dict) or not isinstance(queues.get("items"), list):
        raise unavailable()
    metadata = queues.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("continue"):
        # Never infer no default from an incomplete response.
        raise unavailable()
    defaults = []
    for queue in queues["items"]:
        if not isinstance(queue, dict) or not isinstance(queue.get("metadata"), dict):
            raise unavailable()
        metadata = queue["metadata"]
        annotations = metadata.get("annotations")
        annotations = {} if annotations is None else annotations
        if not isinstance(annotations, dict):
            raise unavailable()
        marker = annotations.get("kueue.x-k8s.io/default-queue", "false")
        if not isinstance(marker, str):
            raise unavailable()
        # Preserve the pinned SDK's case-insensitive annotation convention.
        if marker.lower() == "true":
            name = metadata.get("name")
            if not valid_queue_name(name):
                raise unavailable()
            defaults.append(name)
    if len(defaults) > 1:
        raise unavailable()
    if not defaults:
        return None
    preflight(api, project, defaults[0])
    return str(defaults[0])


def resolve_local_queue(api: client.ApiClient, project: str, queue_name: str | None) -> str | None:
    """Map an API queue selection to the CodeFlare ``local_queue`` for one project.

    A selected queue is returned only after its name, Kueue readiness, project
    eligibility and caller-visible existence in ``project`` are established.
    Without a selection the project's default LocalQueue applies. ``None`` means a
    confirmed unqueued submission, never an unresolved or rejected selection.
    """
    if queue_name is None:
        return default_queue(api, project)
    if not valid_queue_name(queue_name):
        raise ApiError(
            422,
            "validation_error",
            "queue_name must be a DNS subdomain of at most 63 characters.",
        )
    preflight(api, project, queue_name)
    return queue_name
