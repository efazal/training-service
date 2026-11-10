"""Exercise Helm's real schema and templates without a cluster."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from requests.certs import where

CHART = Path(__file__).parents[2] / "charts/training-service"
pytestmark = pytest.mark.skipif(not shutil.which("helm"), reason="helm is not installed")


def render(tmp_path, values):
    path = tmp_path / "values.yaml"
    path.write_text(yaml.safe_dump(values))
    return subprocess.run(
        ["helm", "template", "validation", str(CHART), "-f", str(path)],
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
    role = next(d for d in documents if d["kind"] == "Role")
    assert role["metadata"]["namespace"] == "openshift-config-managed"
    assert role["rules"] == [
        {
            "apiGroups": [""],
            "resources": ["configmaps"],
            "resourceNames": ["default-ingress-cert"],
            "verbs": ["get"],
        }
    ]
    binding = next(d for d in documents if d["kind"] == "RoleBinding")
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
    assert not any(d["kind"] in {"Role", "RoleBinding"} for d in documents)
    pod = next(d for d in documents if d["kind"] == "Deployment")["spec"]["template"]["spec"]
    assert "TRAINING_SERVICE_RAY_CA_DISCOVERY" not in {
        e["name"] for e in pod["containers"][0]["env"]
    }
