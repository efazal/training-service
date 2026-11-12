"""Platform handoff reads only public configuration and produces installable values."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from requests.certs import where
from scripts.platform_values import platform_values


def test_platform_values_auto_discovers_gateway_and_public_trust():
    bundle = Path(where()).read_text()
    read = Mock(
        side_effect=[{"status": {"domain": "gateway.example"}}, {"data": {"ca-bundle.crt": bundle}}]
    )
    values = platform_values(read)
    assert values == {
        "logBackend": {"gatewayURL": "https://gateway.example", "trustedCA": {"bundle": bundle}}
    }
    assert read.call_args_list[0].args[0] == [
        "gatewayconfigs.services.platform.opendatahub.io",
        "default-gateway",
    ]
    assert read.call_args_list[1].args[0] == [
        "configmap",
        "default-ingress-cert",
        "-n",
        "openshift-config-managed",
    ]


def test_platform_values_can_use_normal_trust_without_reading_ca():
    read = Mock(return_value={"status": {"domain": "gateway.example"}})
    assert platform_values(read, system_trust=True) == {
        "logBackend": {"gatewayURL": "https://gateway.example"}
    }
    read.assert_called_once()


@pytest.mark.parametrize(
    "domain",
    [
        "",
        "https://gateway.example",
        "gateway.example/path",
        "user:pass@gateway.example",
        "gateway.example?x=1",
        None,
    ],
)
def test_platform_values_rejects_bad_domains(domain):
    with pytest.raises(ValueError, match="status.domain"):
        platform_values(Mock(return_value={"status": {"domain": domain}}))


@pytest.mark.parametrize("bundle", ["", "invalid", "Fixture PRIVATE KEY marker"])
def test_platform_values_rejects_bad_certificates(bundle):
    with pytest.raises(ValueError, match="certificate"):
        platform_values(
            Mock(
                side_effect=[
                    {"status": {"domain": "gateway.example"}},
                    {"data": {"ca-bundle.crt": bundle}},
                ]
            )
        )
