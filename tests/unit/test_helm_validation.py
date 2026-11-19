"""Exercise Helm's real schema and templates without a cluster."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from requests.certs import where

CHART = Path(__file__).parents[2] / "charts/training-service"
pytestmark = pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")


def test_mlflow_permissions_are_project_scoped_and_server_is_optional(tmp_path):
    result = render(tmp_path, {})
    assert result.returncode == 0, result.stderr
    docs = list(yaml.safe_load_all(result.stdout))
    assert not any(d["kind"] == "MLflow" for d in docs)
    role = next(
        d
        for d in docs
        if d["kind"] == "Role" and d["metadata"]["name"].endswith("-mlflow-training")
    )
    assert role["rules"] == [
        {
            "apiGroups": ["mlflow.kubeflow.org"],
            "resources": ["experiments"],
            "verbs": ["get", "list", "create", "update"],
        }
    ]
    assert not any(
        d["kind"] == "ClusterRoleBinding" and "mlflow-training" in d["metadata"]["name"]
        for d in docs
    )
    binding = next(
        d
        for d in docs
        if d["kind"] == "RoleBinding" and d["metadata"]["name"].endswith("-mlflow-training")
    )
    assert binding["subjects"] == [{"kind": "Group", "name": "system:serviceaccounts:default"}]
    disabled = render(tmp_path, {"mlflow": {"enabled": False}})
    assert disabled.returncode == 0, disabled.stderr
    assert "-mlflow-training" not in disabled.stdout


def test_mlflow_server_creation_requires_installed_operator(tmp_path):
    result = render(tmp_path, {"mlflow": {"server": {"create": True}}})
    assert result.returncode != 0
    assert "Enable the RHOAI MLflow operator" in result.stderr


def test_optional_mlflow_server_has_persistent_storage_and_is_retained():
    result = subprocess.run(
        [
            "helm",
            "template",
            "validation",
            str(CHART),
            "--api-versions",
            "mlflow.opendatahub.io/v1",
            "--set",
            "mlflow.server.create=true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    server = next(d for d in yaml.safe_load_all(result.stdout) if d["kind"] == "MLflow")
    assert server["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"
    assert server["spec"]["storage"]["resources"]["requests"]["storage"] == "10Gi"
    assert server["spec"]["serveArtifacts"] is True


def render(tmp_path, values, namespace="default"):
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump(values))
    return subprocess.run(
        ["helm", "template", "validation", str(CHART), "--namespace", namespace, "-f", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "values, message",
    [
        ({"serviceAccount": {"automount": False}}, "serviceAccount.automount"),
        ({"logBackend": {"trustedCA": {"autoDiscover": "true"}}}, "autoDiscover"),
        (
            {"env": [{"name": "TRAINING_SERVICE_RAY_CA_DISCOVERY", "value": "true"}]},
            "trustedCA.autoDiscover",
        ),
        ({"logBackend": {"gatewayURL": "http://gateway.example"}}, "HTTPS origin"),
        ({"logBackend": {"gatewayURL": "https://user:pass@gateway.example"}}, "HTTPS origin"),
        ({"logBackend": {"gatewayURL": "https://gateway.example/path"}}, "HTTPS origin"),
        ({"logBackend": {"gatewayURL": "https://gateway.example?q=1"}}, "HTTPS origin"),
        ({"logBackend": {"gatewayURL": "https://gateway.example#fragment"}}, "HTTPS origin"),
        ({"logBackend": {"gatewayURL": "https://gateway.example:0"}}, "port"),
        ({"logBackend": {"gatewayURL": "https://gateway.example:99999"}}, "port"),
        (
            {"logBackend": {"gatewayURL": "https://gateway.example", "streamTimeoutSeconds": 0}},
            "streamTimeoutSeconds",
        ),
        (
            {
                "logBackend": {
                    "gatewayURL": "https://gateway.example",
                    "streamTimeoutSeconds": "300",
                }
            },
            "streamTimeoutSeconds",
        ),
        (
            {
                "logBackend": {
                    "gatewayURL": "https://gateway.example",
                    "trustedCA": {"configMapName": "ca", "key": ""},
                }
            },
            "trustedCA.key",
        ),
        (
            {
                "logBackend": {
                    "gatewayURL": "https://gateway.example",
                    "trustedCA": {"configMapName": "ca", "bundle": "PEM"},
                }
            },
            "not both",
        ),
        (
            {
                "logBackend": {
                    "gatewayURL": "https://gateway.example",
                    "trustedCA": {"bundle": "invalid"},
                }
            },
            "public PEM",
        ),
        (
            {
                "logBackend": {"gatewayURL": "https://gateway.example"},
                "env": [
                    {"name": "TRAINING_SERVICE_RAY_GATEWAY_URL", "value": "https://legacy.example"}
                ],
            },
            "not both",
        ),
        (
            {
                "logBackend": {
                    "gatewayURL": "https://gateway.example",
                    "trustedCA": {"configMapName": "ca"},
                },
                "env": [{"name": "TRAINING_SERVICE_RAY_CA_FILE", "value": "/ca.crt"}],
            },
            "not both",
        ),
        (
            {
                "logBackend": {"gatewayURL": "https://gateway.example", "streamTimeoutSeconds": 10},
                "env": [{"name": "TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS", "value": "20"}],
            },
            "not both",
        ),
        (
            {"logBackend": {"gatewayURL": "https://gateway.example"}, "image": {"repository": ""}},
            "repository",
        ),
        ({"logBackend": {"gatewayURL": "https://gateway.example"}, "image": {"tag": ""}}, "tag"),
        (
            {
                "logBackend": {"gatewayURL": "https://gateway.example"},
                "trainingRuntime": {"image": ""},
            },
            "trainingRuntime.image",
        ),
    ],
)
def test_invalid_install_fails_with_actionable_message(tmp_path, values, message):
    result = render(tmp_path, values)
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize(
    "trust", [{}, {"configMapName": "platform-ca"}, {"bundle": Path(where()).read_text()}]
)
def test_chart_mounts_optional_ca_without_subpath_and_sets_timeout(tmp_path, trust):
    result = render(
        tmp_path, {"logBackend": {"gatewayURL": "https://gateway.example", "trustedCA": trust}}
    )
    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    pod = next(d for d in documents if d["kind"] == "Deployment")["spec"]["template"]["spec"]
    container = pod["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert env["TRAINING_SERVICE_RAY_GATEWAY_URL"] == "https://gateway.example"
    assert env["TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS"] == "300"
    if trust:
        assert env["TRAINING_SERVICE_RAY_CA_FILE"] == "/var/run/training-service/log-ca/ca.crt"
        mount = next(m for m in container["volumeMounts"] if m["name"] == "training-service-log-ca")
        assert mount["readOnly"] and "subPath" not in mount
        volume = next(v for v in pod["volumes"] if v["name"] == mount["name"])
        assert volume["configMap"]["items"] == [{"key": "ca-bundle.crt", "path": "ca.crt"}]
        if "bundle" in trust:
            ca = next(
                d for d in documents if d["kind"] == "ConfigMap" and "ca-bundle.crt" in d["data"]
            )
            assert ca["data"]["ca-bundle.crt"].strip() == trust["bundle"].strip()
    else:
        assert "TRAINING_SERVICE_RAY_CA_FILE" not in env
        assert not any(v["name"] == "training-service-log-ca" for v in pod["volumes"])


@pytest.mark.parametrize("token_reviews", [True, False])
def test_default_chart_discovers_gateway_with_only_named_read_access(tmp_path, token_reviews):
    result = render(tmp_path, {"authentication": {"tokenReview": {"enabled": token_reviews}}})
    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    role = next(
        d
        for d in documents
        if d["kind"] == "ClusterRole" and d["metadata"]["name"].endswith("-gateway-discovery")
    )
    assert role["rules"] == [
        {
            "apiGroups": ["services.platform.opendatahub.io"],
            "resources": ["gatewayconfigs"],
            "resourceNames": ["default-gateway"],
            "verbs": ["get"],
        }
    ]
    binding = next(
        d
        for d in documents
        if d["kind"] == "ClusterRoleBinding" and d["metadata"]["name"] == role["metadata"]["name"]
    )
    assert binding["roleRef"]["name"] == role["metadata"]["name"]
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": "validation-training-service", "namespace": "default"}
    ]
    pod = next(d for d in documents if d["kind"] == "Deployment")["spec"]["template"]["spec"]
    assert "TRAINING_SERVICE_RAY_GATEWAY_URL" not in {
        e["name"] for e in pod["containers"][0]["env"]
    }
    assert pod["containers"][0]["readinessProbe"]["timeoutSeconds"] > 5


def test_same_release_in_different_namespaces_has_independent_cluster_rbac(tmp_path):
    names = []
    for namespace in ("team-a", "team-b"):
        result = render(tmp_path, {}, namespace=namespace)
        assert result.returncode == 0, result.stderr
        resources = [
            d
            for d in yaml.safe_load_all(result.stdout)
            if d["kind"] in {"ClusterRole", "ClusterRoleBinding"}
        ]
        names.append({(d["kind"], d["metadata"]["name"]) for d in resources})
        for resource in resources:
            if resource["kind"] == "ClusterRoleBinding":
                assert resource["subjects"][0]["namespace"] == namespace
                assert ("ClusterRole", resource["roleRef"]["name"]) in names[-1]
    assert names[0].isdisjoint(names[1])


@pytest.mark.parametrize("legacy", [True, False])
def test_explicit_gateway_needs_no_platform_discovery_rbac(tmp_path, legacy):
    values = (
        {"env": [{"name": "TRAINING_SERVICE_RAY_GATEWAY_URL", "value": "https://gateway.example"}]}
        if legacy
        else {"logBackend": {"gatewayURL": "https://gateway.example"}}
    )
    result = render(tmp_path, values)
    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    assert not any(d["metadata"]["name"].endswith("-gateway-discovery") for d in documents)


def test_gateway_discovery_binds_custom_service_account(tmp_path):
    result = render(tmp_path, {"serviceAccount": {"create": False, "name": "platform-service"}})
    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    binding = next(
        d
        for d in documents
        if d["kind"] == "ClusterRoleBinding"
        and d["metadata"]["name"].endswith("-gateway-discovery")
    )
    assert binding["subjects"][0]["name"] == "platform-service"


def test_default_install_discovers_ca_with_only_namespace_scoped_named_read_access(tmp_path):
    result = render(tmp_path, {})
    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    role = next(
        d for d in documents if d["kind"] == "Role" and d["metadata"]["name"].endswith("-log-trust")
    )
    assert role["metadata"]["namespace"] == "openshift-config-managed"
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["configmaps"],
            "resourceNames": ["default-ingress-cert"],
            "verbs": ["get"],
        }
    ]
    binding = next(
        d
        for d in documents
        if d["kind"] == "RoleBinding" and d["metadata"]["name"].endswith("-log-trust")
    )
    assert binding["metadata"]["namespace"] == role["metadata"]["namespace"]
    assert binding["roleRef"]["kind"] == "Role"
    assert binding["roleRef"]["name"] == role["metadata"]["name"]
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "validation-training-service",
            "namespace": "default",
        }
    ]
    pod = next(d for d in documents if d["kind"] == "Deployment")["spec"]["template"]["spec"]
    env = {e["name"]: e.get("value") for e in pod["containers"][0]["env"]}
    assert env["TRAINING_SERVICE_RAY_CA_DISCOVERY"] == "true"
    assert "TRAINING_SERVICE_RAY_CA_FILE" not in env
    assert not any(v["name"] == "training-service-log-ca" for v in pod["volumes"])


@pytest.mark.parametrize(
    "values",
    [
        {"logBackend": {"trustedCA": {"autoDiscover": False}}},
        {"logBackend": {"trustedCA": {"configMapName": "customer-ca"}}},
        {"logBackend": {"trustedCA": {"bundle": Path(where()).read_text()}}},
        {"logBackend": {"gatewayURL": "https://custom.example"}},
        {"env": [{"name": "TRAINING_SERVICE_RAY_CA_FILE", "value": "/custom/ca.crt"}]},
    ],
)
def test_custom_trust_or_gateway_omits_platform_ca_permission(tmp_path, values):
    result = render(tmp_path, values)
    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    assert not any(
        d["kind"] in {"Role", "RoleBinding"} and d["metadata"]["name"].endswith("-log-trust")
        for d in documents
    )
    pod = next(d for d in documents if d["kind"] == "Deployment")["spec"]["template"]["spec"]
    assert "TRAINING_SERVICE_RAY_CA_DISCOVERY" not in {
        e["name"] for e in pod["containers"][0]["env"]
    }


def test_kueue_platform_rbac_is_named_read_only_and_namespace_isolated(tmp_path):
    names = []
    for namespace in ("team-a", "team-b"):
        result = render(tmp_path, {"kueuePreflight": {"namespace": namespace}}, namespace=namespace)
        assert result.returncode == 0, result.stderr
        resources = [
            d
            for d in yaml.safe_load_all(result.stdout)
            if d["metadata"]["name"].endswith("-kueue-preflight")
        ]
        assert {d["kind"] for d in resources} == {
            "Role",
            "RoleBinding",
            "ClusterRole",
            "ClusterRoleBinding",
        }
        names.append({(d["kind"], d["metadata"]["name"]) for d in resources})
        for resource in resources:
            if resource["kind"].endswith("Binding"):
                assert resource["subjects"][0]["namespace"] == namespace
                assert (resource["roleRef"]["kind"], resource["roleRef"]["name"]) in names[-1]
            else:
                for rule in resource["rules"]:
                    assert rule["verbs"] == ["get"]
                    if "resources" in rule:
                        assert len(rule["resourceNames"]) == 1
                        assert not set(rule["resources"]) & {"secrets", "localqueues", "rayjobs"}
            if resource["kind"] in {"Role", "RoleBinding"}:
                assert resource["metadata"]["namespace"] == namespace
    assert names[0].isdisjoint(names[1])


def test_kueue_rbac_can_be_provisioned_externally(tmp_path):
    result = render(tmp_path, {"kueuePreflight": {"createRBAC": False}})
    assert result.returncode == 0, result.stderr
    assert not any(
        d["metadata"]["name"].endswith("-kueue-preflight")
        for d in yaml.safe_load_all(result.stdout)
    )


def test_default_offline_install_omits_external_namespace_roles(tmp_path):
    result = render(tmp_path, {})
    assert result.returncode == 0, result.stderr
    resources = [
        d
        for d in yaml.safe_load_all(result.stdout)
        if d["metadata"]["name"].endswith("-kueue-preflight")
    ]
    assert {d["kind"] for d in resources} == {"ClusterRole", "ClusterRoleBinding"}


@pytest.mark.parametrize("namespace", ["true", "false", "null", "123"])
def test_kueue_namespace_fields_are_yaml_strings(tmp_path, namespace):
    result = render(tmp_path, {"kueuePreflight": {"namespace": namespace}}, namespace=namespace)
    assert result.returncode == 0, result.stderr
    resources = [
        d
        for d in yaml.safe_load_all(result.stdout)
        if d["metadata"]["name"].endswith("-kueue-preflight")
    ]
    assert len(resources) == 4
    for resource in resources:
        if resource["kind"] in {"Role", "RoleBinding"}:
            assert resource["metadata"]["namespace"] == namespace
        if resource["kind"].endswith("Binding"):
            assert resource["subjects"][0]["namespace"] == namespace
