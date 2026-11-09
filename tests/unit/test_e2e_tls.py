"""Public trust discovery uses authenticated ConfigMap reads, not TLS bypasses/Secrets."""

import json
import ssl
from argparse import Namespace
from unittest.mock import Mock

import certifi
import pytest
from tests.e2e.environment import prepare_tls


@pytest.fixture
def certificates(monkeypatch):
    monkeypatch.delenv("TRAINING_SERVICE_CA_BUNDLE", raising=False)
    # httpx already supplies certifi; avoid dependence on host-specific OpenSSL root paths.
    # No certificate or key fixture is stored in the repository.
    roots = ssl.create_default_context(cafile=certifi.where()).get_ca_certs(binary_form=True)
    return [ssl.DER_cert_to_PEM_cert(root) for root in roots[:2]]


@pytest.fixture
def ca_values():
    return {
        "env": [{"name": "TRAINING_SERVICE_RAY_CA_FILE", "value": "/trust/ca.crt"}],
        "volumes": [{"name": "ray-ca", "configMap": {"name": "training-service-ray-ca"}}],
        "volumeMounts": [{"name": "ray-ca", "mountPath": "/trust", "readOnly": True}],
    }


def discovery(pem):
    return Mock(return_value=json.dumps({"data": {"ca-bundle.crt": pem}}))


def test_default_discovers_public_ca_for_route_and_ray(tmp_path, certificates, ca_values):
    args = Namespace(ray_ca_bundle=None)
    command = discovery(certificates[0])
    evidence = prepare_tls(args, command, tmp_path, ca_values)
    command.assert_called_once_with(
        [
            "oc",
            "get",
            "configmap",
            "default-ingress-cert",
            "-n",
            "openshift-config-managed",
            "-o",
            "json",
        ]
    )
    assert args.ingress_ca_bundle == tmp_path / "ingress-ca.pem"
    assert args.ray_ca_bundle == args.ingress_ca_bundle
    assert args.ingress_ca_bundle.read_text() == certificates[0]
    assert evidence["ingress_ca_source"] == "openshift-config-managed/default-ingress-cert"
    assert evidence["ray_ca_source"] == evidence["ingress_ca_source"]
    assert len(evidence["ingress_ca_sha256"]) == 64
    ssl.create_default_context(cafile=args.ingress_ca_bundle)


def test_ray_override_is_not_applied_to_route(tmp_path, certificates, ca_values):
    override = tmp_path / "custom-ray.pem"
    override.write_text(certificates[-1])
    args = Namespace(ray_ca_bundle=override)
    evidence = prepare_tls(args, discovery(certificates[0]), tmp_path, ca_values)
    assert args.ray_ca_bundle == override
    assert args.ingress_ca_bundle != override
    assert args.ingress_ca_bundle.read_text() == certificates[0]
    assert evidence["ray_ca_source"] == "--ray-ca-bundle"


def test_ingress_override_is_not_applied_to_ray(monkeypatch, tmp_path, certificates, ca_values):
    override = tmp_path / "custom-route.pem"
    override.write_text(certificates[-1])
    monkeypatch.setenv("TRAINING_SERVICE_CA_BUNDLE", str(override))
    args = Namespace(ray_ca_bundle=None)
    evidence = prepare_tls(args, discovery(certificates[0]), tmp_path, ca_values)
    assert args.ingress_ca_bundle == override
    assert args.ray_ca_bundle != override
    assert args.ray_ca_bundle.read_text() == certificates[0]
    assert evidence["ingress_ca_source"] == "TRAINING_SERVICE_CA_BUNDLE"


def test_explicit_overrides_do_not_need_configmap_access(
    monkeypatch, tmp_path, certificates, ca_values
):
    override = tmp_path / "explicit-ca.pem"
    override.write_text(certificates[0])
    monkeypatch.setenv("TRAINING_SERVICE_CA_BUNDLE", str(override))
    args = Namespace(ray_ca_bundle=override)
    command = Mock()
    evidence = prepare_tls(args, command, tmp_path, ca_values)
    command.assert_not_called()
    assert args.ingress_ca_bundle == args.ray_ca_bundle == override
    assert not (tmp_path / "ingress-ca.pem").exists()
    assert evidence["ray_ca_source"] == "--ray-ca-bundle"


def test_public_gateway_can_keep_system_trust(tmp_path, certificates):
    args = Namespace(ray_ca_bundle=None)
    evidence = prepare_tls(args, discovery(certificates[0]), tmp_path, {})
    assert args.ray_ca_bundle is None
    assert args.ingress_ca_bundle.is_file()
    assert evidence["ray_ca_source"] == "system trust"


@pytest.mark.parametrize("pem", [None, "", "not a certificate", "Fixture PRIVATE KEY marker"])
def test_bad_discovered_data_is_not_saved(tmp_path, certificates, ca_values, pem):
    with pytest.raises(ValueError, match="public certificates|valid PEM|private keys"):
        prepare_tls(Namespace(ray_ca_bundle=None), discovery(pem), tmp_path, ca_values)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("payload", ["not json", "{}"])
def test_missing_or_bad_configmap_has_actionable_failure(
    tmp_path, certificates, ca_values, payload
):
    with pytest.raises(ValueError, match="ConfigMap|public certificates"):
        prepare_tls(Namespace(ray_ca_bundle=None), Mock(return_value=payload), tmp_path, ca_values)
    assert not list(tmp_path.iterdir())


def test_denied_discovery_fails_closed(tmp_path, certificates, ca_values):
    command = Mock(side_effect=RuntimeError("raw command error withheld"))
    with pytest.raises(ValueError, match="Grant get access.*TLS verification is never disabled"):
        prepare_tls(Namespace(ray_ca_bundle=None), command, tmp_path, ca_values)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("target", ["ray", "ingress"])
def test_private_key_overrides_are_rejected_before_discovery(
    monkeypatch, tmp_path, certificates, ca_values, target
):
    override = tmp_path / "not-a-public-bundle.txt"
    override.write_text("Fixture PRIVATE KEY marker, not cryptographic material.\n")
    args = Namespace(ray_ca_bundle=override if target == "ray" else None)
    if target == "ingress":
        monkeypatch.setenv("TRAINING_SERVICE_CA_BUNDLE", str(override))
    command = Mock()
    with pytest.raises(ValueError, match="never private keys"):
        prepare_tls(args, command, tmp_path, ca_values)
    command.assert_not_called()


def test_ca_mount_must_remain_readonly(tmp_path, certificates, ca_values):
    ca_values["volumeMounts"][0]["readOnly"] = False
    command = Mock()
    with pytest.raises(ValueError, match="read-only"):
        prepare_tls(Namespace(ray_ca_bundle=None), command, tmp_path, ca_values)
    command.assert_not_called()


def test_structured_helm_bundle_needs_no_duplicate_ray_configmap(tmp_path, certificates):
    args = Namespace(ray_ca_bundle=None)
    values = {"logBackend": {"trustedCA": {"bundle": certificates[-1]}}}
    evidence = prepare_tls(args, discovery(certificates[0]), tmp_path, values)
    assert args.ray_ca_bundle is None
    assert evidence["ray_ca_source"] == "Helm public CA bundle"
    assert args.ingress_ca_bundle.read_text() == certificates[0]


def test_structured_helm_bundle_rejects_legacy_override(tmp_path, certificates, ca_values):
    ca_values["logBackend"] = {"trustedCA": {"bundle": certificates[0]}}
    with pytest.raises(ValueError, match="not both"):
        prepare_tls(Namespace(ray_ca_bundle=None), Mock(), tmp_path, ca_values)
