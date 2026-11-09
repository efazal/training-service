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
        ({}, "logBackend.gatewayURL is required"),
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
