"""Capability checks use platform identity; namespace and queues use caller identity."""

from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from training_service import kueue_preflight as k
from training_service.errors import ApiError

_REAL_SERVED_VERSION = k.served_version


@pytest.mark.parametrize("version", ["v1beta1", "v1beta2", "v1beta3"])
def test_served_versions(monkeypatch, version):
    api = Mock()
    api.call_api.return_value = {
        "resources": [{"name": "localqueues", "namespaced": True, "verbs": ["get", "list"]}]
    }
    groups = Mock()
    groups.get_api_versions.return_value = NS(
        groups=[
            NS(name=k.GROUP, versions=[NS(version=version)], preferred_version=NS(version=version))
        ]
    )
    monkeypatch.setattr(client, "ApisApi", lambda _: groups)
    assert k.served_version(api) == version
    assert api.call_api.call_args.args[0] == f"/apis/{k.GROUP}/{version}"


@pytest.fixture
def installation(monkeypatch):
    api = Mock(sanitize_for_serialization=lambda obj: obj)

    @contextmanager
    def platform():
        yield api

    monkeypatch.setattr(k, "platform_client", platform)
    monkeypatch.setattr(k, "served_version", lambda _: "v1beta1")
    dep = NS(
        metadata=NS(generation=1),
        status=NS(
            observed_generation=1,
            available_replicas=1,
            updated_replicas=1,
            ready_replicas=1,
            replicas=1,
        ),
        spec=NS(
            replicas=1,
            template=NS(
                metadata=NS(labels={"app": "kueue"}, annotations={}),
                spec=NS(
                    volumes=[NS(name="config", config_map=NS(name="kueue-manager-config"))],
                    containers=[
                        NS(
                            args=["--config=/etc/controller_manager_config.yaml"],
                            volume_mounts=[NS(name="config", mount_path="/etc", sub_path=None)],
                        )
                    ],
                ),
            ),
        ),
    )
    apps = Mock()
    apps.read_namespaced_deployment.return_value = dep
    core = Mock()
    core.read_namespaced_config_map.return_value = NS(
        data={
            "controller_manager_config.yaml": "integrations:\n  frameworks: [ray.io/rayjob]\n"
            "managedJobsNamespaceSelector:\n  matchLabels:\n"
            '    kueue.openshift.io/managed: "true"\n'
        }
    )
    core.read_namespaced_service.return_value = NS(
        spec=NS(
            selector={"app": "kueue"},
            ports=[NS(name="https", port=443, target_port=9443, protocol="TCP")],
        )
    )
    core.read_namespaced_endpoints.return_value = NS(
        subsets=[
            NS(addresses=[NS(ip="127.0.0.1")], ports=[NS(name="https", port=9443, protocol="TCP")])
        ]
    )
    webhook = NS(
        client_config=NS(
            service=NS(
                namespace="openshift-kueue-operator",
                name="kueue-webhook-service",
                path="/mutate-ray-io-v1-rayjob",
                port=None,
            )
        ),
        failure_policy="Fail",
        match_conditions=None,
        object_selector=None,
        namespace_selector={},
        rules=[
            NS(
                api_groups=["ray.io"],
                api_versions=["v1"],
                resources=["rayjobs"],
                operations=["CREATE"],
                scope=None,
            )
        ],
    )
    admission = Mock()
    admission.read_mutating_webhook_configuration.return_value = NS(webhooks=[webhook])
    monkeypatch.setattr(client, "AppsV1Api", lambda _: apps)
    monkeypatch.setattr(client, "CoreV1Api", lambda _: core)
    monkeypatch.setattr(client, "AdmissionregistrationV1Api", lambda _: admission)
    return dep, core, webhook


def test_separate_installation_without_dsc(installation):
    assert k.capability({"kueue.openshift.io/managed": "true"}) == "v1beta1"


def test_opt_out_with_retained_queues(installation, monkeypatch):
    custom = Mock()
    monkeypatch.setattr(client, "CustomObjectsApi", lambda _: custom)
    installation[1].read_namespace.return_value = NS(metadata=NS(labels={}))
    with pytest.raises(ApiError) as error:
        k.preflight(Mock(), "project", "retained-queue")
    assert (error.value.status_code, error.value.code) == (409, "kueue_project_excluded")
    custom.get_namespaced_custom_object.assert_not_called()


def test_stale_crds_unavailable_controller(installation):
    installation[0].status.available_replicas = 0
    with pytest.raises(ApiError) as error:
        k.capability({"kueue.openshift.io/managed": "true"})
    assert error.value.status_code == 503


@pytest.mark.parametrize("status", [401, 403, 404])
def test_selected_queue_checked_with_caller_on_each_post(monkeypatch, status, installation):
    api, core, custom = Mock(), Mock(), Mock()
    core.read_namespace.return_value = NS(metadata=NS(labels={}))
    monkeypatch.setattr(client, "CoreV1Api", lambda passed: core if passed is api else None)
    monkeypatch.setattr(
        client, "CustomObjectsApi", lambda passed: custom if passed is api else None
    )
    monkeypatch.setattr(k, "capability", lambda _: "v1beta1")
    monkeypatch.setattr(k, "serves_localqueues", lambda *_: True)
    custom.get_namespaced_custom_object.side_effect = ApiException(status=status)
    with pytest.raises(ApiError if status == 404 else ApiException):
        k.preflight(api, "project", "selected")
    assert custom.get_namespaced_custom_object.call_args.args == (
        k.GROUP,
        "v1beta1",
        "project",
        "localqueues",
        "selected",
    )


@pytest.mark.parametrize(
    "op,values,expected",
    [
        ("In", ["a"], True),
        ("NotIn", ["a"], False),
        ("Exists", [], True),
        ("DoesNotExist", [], False),
    ],
)
def test_selector_expressions(op, values, expected):
    assert (
        k.matches(
            {"matchExpressions": [{"key": "team", "operator": op, "values": values}]}, {"team": "a"}
        )
        is expected
    )


def test_absence_requires_successful_discovery(monkeypatch):
    discovery = Mock()
    discovery.get_api_versions.return_value = NS(groups=[])
    monkeypatch.setattr(client, "ApisApi", lambda _: discovery)
    with pytest.raises(ApiError) as error:
        k.served_version(Mock())
    assert error.value.code == "kueue_not_installed"


@pytest.mark.parametrize("status", [403, 500])
def test_platform_check_failure_is_not_disabled(monkeypatch, status):
    monkeypatch.setattr(k.config, "load_incluster_config", lambda **_: None)
    with pytest.raises(ApiError) as error:
        with k.platform_client():
            raise ApiException(status=status, reason="private")
    assert error.value.status_code == 503
    assert "private" not in error.value.message


@pytest.mark.parametrize(
    "selector",
    [
        client.V1LabelSelector(),
        client.V1LabelSelector(match_labels={}),
        client.V1LabelSelector(match_expressions=[]),
    ],
)
def test_empty_kubernetes_selector_models(installation, monkeypatch, selector):
    installation[2].object_selector = selector

    @contextmanager
    def platform():
        with client.ApiClient() as api:
            yield api

    monkeypatch.setattr(k, "platform_client", platform)
    assert k.capability({"kueue.openshift.io/managed": "true"}) == "v1beta1"


@pytest.mark.parametrize(
    "selector",
    [
        {},
        {"matchLabels": {}},
        {"matchExpressions": []},
        {"matchLabels": {}, "matchExpressions": []},
    ],
)
def test_equivalent_empty_object_selectors(installation, selector):
    installation[2].object_selector = selector
    assert k.capability({"kueue.openshift.io/managed": "true"}) == "v1beta1"


@pytest.mark.parametrize(
    "selector",
    [{"matchLabels": {"team": "a"}}, {"matchExpressions": [{"key": "team", "operator": "Exists"}]}],
)
def test_filtered_object_selectors_fail_closed(installation, selector):
    installation[2].object_selector = selector
    with pytest.raises(ApiError) as error:
        k.capability({"kueue.openshift.io/managed": "true"})
    assert error.value.status_code == 503


def test_healthy_partial_rollout_without_packaging_checks(installation):
    dep = installation[0]
    dep.spec.replicas = 2
    dep.status.available_replicas = 1
    dep.status.updated_replicas = 1
    dep.status.ready_replicas = 1
    dep.status.replicas = 2
    dep.status.observed_generation = 0
    dep.spec.template.metadata.annotations = {}
    dep.spec.template.spec.volumes = []
    dep.spec.template.spec.containers = []
    assert k.capability({"kueue.openshift.io/managed": "true"}) == "v1beta1"


def test_no_ready_webhook_backend_fails_closed(installation):
    installation[1].read_namespaced_endpoints.return_value = NS(subsets=[])
    with pytest.raises(ApiError) as error:
        k.capability({"kueue.openshift.io/managed": "true"})
    assert error.value.status_code == 503


@pytest.mark.parametrize(
    "selector",
    [
        {"matchLabels": None},
        {"matchExpressions": None},
        {"matchLabels": None, "matchExpressions": None},
    ],
)
def test_null_selector_fields_match_empty(selector):
    assert k.matches(selector, {"team": "a"}) is True


@pytest.mark.parametrize(
    "selector",
    [
        {"matchLabels": []},
        {"matchExpressions": {}},
        {"matchExpressions": [{"key": "team", "operator": ["In"], "values": ["a"]}]},
        {"matchExpressions": [{"key": "team", "operator": {}, "values": []}]},
    ],
)
def test_malformed_selectors_fail_with_503(selector):
    with pytest.raises(ApiError) as error:
        k.matches(selector, {})
    assert (error.value.status_code, error.value.code) == (503, "backend_unavailable")


@pytest.mark.parametrize(
    "resources",
    [
        None,
        {"resources": [None]},
        {"resources": [{"name": "localqueues", "namespaced": True, "verbs": None}]},
        {"resources": [{"name": "localqueues", "namespaced": True, "verbs": ["get", {}]}]},
    ],
)
def test_malformed_discovery_fails_with_503(resources):
    api = Mock()
    api.call_api.return_value = resources
    with pytest.raises(ApiError) as error:
        k.serves_localqueues(api, "v1beta2")
    assert error.value.status_code == 503


@pytest.mark.parametrize("frameworks", ["ray.io/rayjob-disabled", 1, None, [1], {}, []])
def test_malformed_or_missing_ray_integration_fails_closed(installation, frameworks):
    import yaml

    core = installation[1]
    core.read_namespaced_config_map.return_value.data["controller_manager_config.yaml"] = (
        yaml.safe_dump(
            {
                "integrations": {"frameworks": frameworks},
                "managedJobsNamespaceSelector": {},
            }
        )
    )
    with pytest.raises(ApiError) as error:
        k.capability({})
    assert error.value.status_code == 503


@pytest.mark.parametrize(
    "scope,operations,accepted",
    [
        ("Cluster", ["CREATE"], False),
        ("Namespaced", ["CREATE"], True),
        ("*", ["CREATE"], True),
        (None, ["*"], True),
        ("Namespaced", ["*"], True),
        ("Namespaced", ["UPDATE"], False),
    ],
)
def test_namespaced_create_rule_matching(installation, scope, operations, accepted):
    rule = installation[2].rules[0]
    rule.scope, rule.operations = scope, operations
    if accepted:
        assert k.capability({"kueue.openshift.io/managed": "true"}) == "v1beta1"
    else:
        with pytest.raises(ApiError) as error:
            k.capability({"kueue.openshift.io/managed": "true"})
        assert error.value.status_code == 503


@pytest.mark.parametrize(
    "change,accepted",
    [
        ("wrong_service_port", False),
        ("wrong_endpoint_port", False),
        ("wrong_endpoint_name", False),
        ("wrong_protocol", False),
        ("named_target", True),
        ("custom_port", True),
        ("not_ready", False),
    ],
)
def test_readiness_matches_webhook_service_port(installation, change, accepted):
    _, core, webhook = installation
    port = core.read_namespaced_service.return_value.spec.ports[0]
    endpoint = core.read_namespaced_endpoints.return_value.subsets[0]
    if change == "wrong_service_port":
        port.port = 8443
    elif change == "wrong_endpoint_port":
        endpoint.ports[0].port = 8443
    elif change == "wrong_endpoint_name":
        endpoint.ports[0].name = "metrics"
    elif change == "wrong_protocol":
        endpoint.ports[0].protocol = "UDP"
    elif change == "named_target":
        port.target_port = "webhook-container"
    elif change == "custom_port":
        webhook.client_config.service.port = 8443
        port.port = 8443
    elif change == "not_ready":
        endpoint.addresses = []
    if accepted:
        assert k.capability({"kueue.openshift.io/managed": "true"}) == "v1beta1"
    else:
        with pytest.raises(ApiError) as error:
            k.capability({"kueue.openshift.io/managed": "true"})
        assert error.value.status_code == 503


@pytest.mark.parametrize("served,status", [(True, 404), (False, 503), ("removed", 503)])
def test_queue_404_rechecks_exact_version(installation, monkeypatch, served, status):
    custom = Mock()
    custom.get_namespaced_custom_object.side_effect = ApiException(status=404)
    monkeypatch.setattr(client, "CustomObjectsApi", lambda _: custom)
    probe = Mock(return_value=served is True)
    if served == "removed":
        probe.side_effect = k.unavailable()
    monkeypatch.setattr(k, "serves_localqueues", probe)
    with pytest.raises(ApiError) as error:
        k.check_queue(Mock(), "project", "queue", "v1beta2")
    assert error.value.status_code == status
    assert probe.call_args.args[1] == "v1beta2"


@pytest.fixture
def default_discovery(installation, monkeypatch):
    caller = Mock()
    custom = Mock()
    custom.list_namespaced_custom_object.return_value = {"items": []}
    monkeypatch.setattr(
        client, "CustomObjectsApi", lambda passed: custom if passed is caller else None
    )
    installation[1].read_namespace.return_value = NS(
        metadata=NS(labels={"kueue.openshift.io/managed": "true"})
    )
    return caller, custom


def default_item(name="gpu.batch"):
    return {"metadata": {"name": name, "annotations": {"kueue.x-k8s.io/default-queue": "true"}}}


@pytest.mark.parametrize("version", ["v1beta2", "v1beta3"])
def test_default_queue_uses_only_served_version(
    installation, default_discovery, monkeypatch, version
):
    caller, custom = default_discovery
    groups = Mock()
    groups.get_api_versions.return_value = NS(
        groups=[
            NS(name=k.GROUP, versions=[NS(version=version)], preferred_version=NS(version=version))
        ]
    )
    monkeypatch.setattr(client, "ApisApi", lambda _: groups)
    monkeypatch.setattr(k, "served_version", _REAL_SERVED_VERSION)
    with k.platform_client() as api:
        api.call_api.return_value = {
            "resources": [{"name": "localqueues", "namespaced": True, "verbs": ["get", "list"]}]
        }
    custom.list_namespaced_custom_object.return_value = {"items": [default_item()]}
    assert k.default_queue(caller, "project") == "gpu.batch"
    assert custom.list_namespaced_custom_object.call_args.args[1] == version
    assert custom.get_namespaced_custom_object.call_args.args[1] == version


def test_default_absence_allows_unqueued(default_discovery):
    caller, custom = default_discovery
    assert k.default_queue(caller, "project") is None
    custom.get_namespaced_custom_object.assert_not_called()


def test_absent_kueue_allows_unqueued(default_discovery, monkeypatch):
    caller, custom = default_discovery
    monkeypatch.setattr(
        k, "served_version", Mock(side_effect=ApiError(409, "kueue_not_installed", "Absent"))
    )
    assert k.default_queue(caller, "project") is None
    custom.list_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"items": None},
        {"items": [None]},
        {"items": [], "metadata": {"continue": "next"}},
        {"items": [default_item(), default_item("other")]},
        {"items": [default_item("bad/name")]},
        {"items": [default_item("a" * 64)]},
    ],
)
def test_uncertain_defaults_do_not_allow_unqueued(default_discovery, response):
    caller, custom = default_discovery
    custom.list_namespaced_custom_object.return_value = response
    with pytest.raises(ApiError) as error:
        k.default_queue(caller, "project")
    assert error.value.status_code == 503


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_default_lookup_errors_never_fall_back_to_unqueued(default_discovery, status):
    caller, custom = default_discovery
    custom.list_namespaced_custom_object.side_effect = ApiException(status=status)
    with pytest.raises(ApiException if status in (401, 403) else ApiError) as error:
        k.default_queue(caller, "project")
    assert (
        getattr(error.value, "status", None) == status
        if status in (401, 403)
        else error.value.status_code == 503
    )


def test_default_queue_respects_project_opt_out(installation, default_discovery):
    caller, custom = default_discovery
    installation[1].read_namespace.return_value.metadata.labels = {}
    custom.list_namespaced_custom_object.return_value = {"items": [default_item()]}
    with pytest.raises(ApiError) as error:
        k.default_queue(caller, "project")
    assert error.value.code == "kueue_project_excluded"
    custom.get_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("marker", ["true", "TRUE", "True"])
def test_default_queue_annotation_preserves_case_insensitive_sdk_semantics(
    default_discovery, marker
):
    caller, custom = default_discovery
    item = default_item()
    item["metadata"]["annotations"]["kueue.x-k8s.io/default-queue"] = marker
    custom.list_namespaced_custom_object.return_value = {"items": [item]}
    assert k.default_queue(caller, "project") == "gpu.batch"
    custom.get_namespaced_custom_object.assert_called_once()


@pytest.mark.parametrize("marker", [None, True, 1, ["true"], {}])
def test_non_string_default_queue_annotation_fails_closed(default_discovery, marker):
    caller, custom = default_discovery
    item = default_item()
    item["metadata"]["annotations"]["kueue.x-k8s.io/default-queue"] = marker
    custom.list_namespaced_custom_object.return_value = {"items": [item]}
    with pytest.raises(ApiError) as error:
        k.default_queue(caller, "project")
    assert (error.value.status_code, error.value.code) == (503, "backend_unavailable")
    custom.get_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("version", ["v1beta2", "v1beta3"])
def test_post_without_queue_uses_real_discovery_and_preflight_on_non_beta1_cluster(
    installation, default_discovery, monkeypatch, version
):
    from fastapi.testclient import TestClient

    from training_service import jobs
    from training_service.auth import CallerIdentity
    from training_service.contract import load_contract
    from training_service.main import create_app

    caller, custom = default_discovery
    groups = Mock()
    groups.get_api_versions.return_value = NS(
        groups=[
            NS(name=k.GROUP, versions=[NS(version=version)], preferred_version=NS(version=version))
        ]
    )
    monkeypatch.setattr(client, "ApisApi", lambda _: groups)
    monkeypatch.setattr(k, "served_version", _REAL_SERVED_VERSION)
    with k.platform_client() as platform:
        platform.call_api.return_value = {
            "resources": [{"name": "localqueues", "namespaced": True, "verbs": ["get", "list"]}]
        }
    custom.list_namespaced_custom_object.return_value = {"items": [default_item()]}
    installation[1].read_namespaced_persistent_volume_claim.return_value = NS(
        spec=NS(access_modes=["ReadWriteMany"])
    )

    @contextmanager
    def caller_api(token):
        assert token == "caller-token"
        yield caller

    monkeypatch.setattr(jobs, "_caller_api_client", caller_api)
    examples = load_contract()["paths"]["/projects/{project}/training-jobs"]["post"]["requestBody"][
        "content"
    ]["application/json"]["examples"]
    body = dict(examples["pvc_inputs_output"]["value"])
    body.pop("queue_name", None)
    sdk = Mock()
    sdk.return_value.submit.return_value = body["name"]
    monkeypatch.setattr(jobs, "_MetadataRayJob", sdk)

    class Authenticator:
        async def authenticate(self, token):
            assert token == "caller-token"
            return CallerIdentity("alice", "uid-a", ())

    class Authorizer:
        async def is_allowed(self, identity, project, permission):
            assert identity.username == "alice"
            assert project == "project"
            assert (permission.group, permission.resource, permission.verb) == (
                "ray.io",
                "rayjobs",
                "create",
            )
            return True

    http = TestClient(
        create_app(
            Authenticator(),
            Authorizer(),
            backend=jobs.CodeFlareJobBackend("quay.io/example/ray:2.58.0"),
        )
    )
    response = http.post(
        "/api/v1/projects/project/training-jobs",
        headers={"Authorization": "Bearer caller-token"},
        json=body,
    )
    assert response.status_code == 202, response.json()
    assert "queue_name" not in body
    assert custom.list_namespaced_custom_object.call_args.args == (
        k.GROUP,
        version,
        "project",
        "localqueues",
    )
    assert custom.get_namespaced_custom_object.call_args.args == (
        k.GROUP,
        version,
        "project",
        "localqueues",
        "gpu.batch",
    )
    assert all(
        call.args[0] == f"/apis/{k.GROUP}/{version}" for call in platform.call_api.call_args_list
    )
    assert sdk.call_args.kwargs["local_queue"] == "gpu.batch"
    assert sdk.call_args.kwargs["api_client"] is caller
    assert sdk.call_args.kwargs["annotations"]["training.rhoai.io/submitted-by"] == "alice"
    sdk.return_value.submit.assert_called_once()


def test_selected_queue_maps_to_local_queue_in_requested_project(default_discovery, installation):
    caller, custom = default_discovery
    assert k.resolve_local_queue(caller, "project", "gpu.batch") == "gpu.batch"
    assert installation[1].read_namespace.call_args.args == ("project",)
    assert custom.get_namespaced_custom_object.call_args.args == (
        k.GROUP,
        "v1beta1",
        "project",
        "localqueues",
        "gpu.batch",
    )
    custom.list_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("name", ["", "Bad", "bad/name", "a..b", "-edge", "a" * 64, 7])
def test_invalid_queue_selection_is_rejected_before_cluster_access(
    default_discovery, installation, name
):
    caller, custom = default_discovery
    with pytest.raises(ApiError) as error:
        k.resolve_local_queue(caller, "project", name)
    assert (error.value.status_code, error.value.code) == (422, "validation_error")
    # Namespace and platform reads go through the patched core API, not `caller`.
    assert installation[1].mock_calls == [] and custom.mock_calls == []


@pytest.mark.parametrize("status", [0, 400, 409, 410, 429, 500, 503])
def test_failed_queue_lookup_is_uncertain_not_missing(default_discovery, status):
    caller, custom = default_discovery
    custom.get_namespaced_custom_object.side_effect = ApiException(status=status)
    with pytest.raises(ApiError) as error:
        k.resolve_local_queue(caller, "project", "gpu.batch")
    assert (error.value.status_code, error.value.code) == (503, "backend_unavailable")


def test_queue_missing_from_requested_project_is_rejected(default_discovery, monkeypatch):
    caller, custom = default_discovery
    monkeypatch.setattr(k, "serves_localqueues", lambda *_: True)
    custom.get_namespaced_custom_object.side_effect = ApiException(status=404)
    with pytest.raises(ApiError) as error:
        k.resolve_local_queue(caller, "project", "other-project-queue")
    assert (error.value.status_code, error.value.code) == (404, "not_found")
    assert "project 'project'" in error.value.message


@pytest.mark.parametrize(
    "labels,absent,expected",
    [
        ({}, False, (409, "kueue_project_excluded")),
        ({"kueue.openshift.io/managed": "true"}, True, (409, "kueue_not_installed")),
    ],
)
def test_unavailable_queue_selection_is_rejected(
    default_discovery, installation, monkeypatch, labels, absent, expected
):
    caller, custom = default_discovery
    installation[1].read_namespace.return_value = NS(metadata=NS(labels=labels))
    if absent:
        monkeypatch.setattr(
            k, "served_version", Mock(side_effect=ApiError(409, "kueue_not_installed", "Absent"))
        )
    with pytest.raises(ApiError) as error:
        k.resolve_local_queue(caller, "project", "gpu.batch")
    assert (error.value.status_code, error.value.code) == expected
    custom.get_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("items,expected", [([], None), ([default_item()], "gpu.batch")])
def test_omitted_queue_selection_uses_project_default(default_discovery, items, expected):
    caller, custom = default_discovery
    custom.list_namespaced_custom_object.return_value = {"items": items}
    assert k.resolve_local_queue(caller, "project", None) == expected
    assert custom.list_namespaced_custom_object.call_args.args[2] == "project"
