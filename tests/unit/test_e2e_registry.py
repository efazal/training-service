"""Registry auth safety proofs use synthetic credentials, never the real auth store."""

import base64
import json
import logging
import subprocess
from unittest.mock import MagicMock, Mock

import pytest
from kubernetes import client
from kubernetes.client.exceptions import ApiException
from tests.e2e import deploy, registry


@pytest.fixture
def authfile(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    credential = base64.b64encode(b"test-robot:synthetic-secret").decode()
    path.write_text(
        json.dumps(
            {
                "auths": {
                    "quay.io": {"auth": credential, "email": "unused@example.com"},
                    "unrelated.invalid": {"auth": "must-not-be-copied"},
                }
            }
        )
    )
    monkeypatch.setenv("REGISTRY_AUTH_FILE", str(path))
    return path, credential


def test_only_quay_credential_is_exported_without_local_writes(authfile):
    path, credential = authfile
    before = path.read_bytes()
    assert registry.quay_auth() == {"auths": {"quay.io": {"auth": credential}}}
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    "document",
    [
        "not-json",
        "[]",
        '{"auths":null}',
        '{"auths":{"quay.io":{"auth":"invalid"}}}',
        json.dumps({"auths": {"quay.io": {"auth": base64.b64encode(b"name:").decode()}}}),
        json.dumps({"credHelpers": {"quay.io": "../../unsafe-helper"}}),
    ],
)
def test_invalid_auth_is_actionable_and_redacted(authfile, document):
    path, _ = authfile
    path.write_text(document)
    with pytest.raises(ValueError, match="podman login") as error:
        registry.quay_auth()
    assert "synthetic-secret" not in str(error.value)
    assert "invalid" not in str(error.value)


def test_explicit_auth_file_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("REGISTRY_AUTH_FILE", str(tmp_path / "missing.json"))
    assert registry.auth_paths() == [tmp_path / "missing.json"]
    with pytest.raises(ValueError, match="No Quay credentials"):
        registry.quay_auth()


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_auth_discovery_uses_podman_locations(monkeypatch, tmp_path, platform):
    monkeypatch.delenv("REGISTRY_AUTH_FILE", raising=False)
    monkeypatch.setattr(registry.sys, "platform", platform)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "configuration"))
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "compatibility"))
    expected = [
        tmp_path / "configuration/containers/auth.json",
        tmp_path / "compatibility/config.json",
    ]
    if platform == "linux":
        expected.insert(0, tmp_path / "runtime/containers/auth.json")
    assert registry.auth_paths() == expected


@pytest.mark.parametrize("store", ["credHelpers", "credsStore"])
def test_native_credential_helper_is_captured_not_logged(authfile, monkeypatch, store, capsys):
    path, credential = authfile
    path.write_text(
        json.dumps({store: {"quay.io": "test-helper"} if store == "credHelpers" else "test-helper"})
    )
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "Username": "test-robot",
                    "Secret": "synthetic-secret",
                }
            ),
            "",
        )
    )
    monkeypatch.setattr(registry.subprocess, "run", run)
    assert registry.quay_auth() == {"auths": {"quay.io": {"auth": credential}}}
    assert run.call_args.args[0] == ["docker-credential-test-helper", "get"]
    assert run.call_args.kwargs["input"] == "quay.io\n"
    assert run.call_args.kwargs["capture_output"]
    assert capsys.readouterr().out == ""


def test_helper_failure_never_exposes_output(authfile, monkeypatch):
    path, _ = authfile
    path.write_text('{"credHelpers":{"quay.io":"test-helper"}}')
    monkeypatch.setattr(
        registry.subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [],
                1,
                "synthetic-secret",
                "synthetic-secret",
            )
        ),
    )
    with pytest.raises(ValueError) as error:
        registry.quay_auth()
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize("verified", [True, False])
def test_registry_transfer_refuses_debug_logging(monkeypatch, verified):
    def load(client_configuration):
        client_configuration.verify_ssl = verified
        client_configuration.debug = True

    monkeypatch.setattr(registry.config, "load_kube_config", load)
    auth = Mock()
    monkeypatch.setattr(registry, "quay_auth", auth)
    with pytest.raises(ValueError, match="debug logging"):
        registry.preflight()
    auth.assert_not_called()


def test_registry_transfer_refuses_inherited_http_debug_logging(monkeypatch, caplog):
    monkeypatch.setattr(registry.config, "load_kube_config", Mock())
    auth = Mock()
    monkeypatch.setattr(registry, "quay_auth", auth)
    with caplog.at_level(logging.DEBUG, logger="kubernetes.client.rest"):
        with pytest.raises(ValueError, match="debug logging"):
            registry.preflight()
    auth.assert_not_called()


@pytest.mark.parametrize("verified", [True, False])
def test_registry_transfer_honors_existing_kubeconfig_and_warns(monkeypatch, capsys, verified):
    def load(client_configuration):
        client_configuration.verify_ssl = verified

    monkeypatch.setattr(registry.config, "load_kube_config", load)
    version = Mock()
    monkeypatch.setattr(registry.client, "VersionApi", Mock(return_value=version))
    auth = Mock()
    monkeypatch.setattr(registry, "quay_auth", auth)
    with registry.kubeconfig_client() as api:
        assert api.configuration.verify_ssl is verified
    registry.preflight()
    version.get_code.assert_called_once()
    auth.assert_called_once()
    output = capsys.readouterr().out
    assert ("WARNING:" in output) == (not verified)
    if not verified:
        assert "without verifying API-server identity" in output


@pytest.fixture
def secret_api(monkeypatch):
    monkeypatch.setattr(registry, "kubeconfig_client", MagicMock())
    core = Mock()
    core.read_namespace.return_value = client.V1Namespace(
        metadata=client.V1ObjectMeta(
            uid="namespace-uid",
            labels={registry.RUN_LABEL: "unit-run"},
        )
    )
    metadata = client.V1ObjectMeta(
        uid="secret-uid",
        labels={registry.RUN_LABEL: "unit-run"},
        owner_references=[
            client.V1OwnerReference(
                api_version="v1", kind="Namespace", name="unit-ns", uid="namespace-uid"
            )
        ],
    )
    core.create_namespaced_secret.return_value = client.V1Secret(metadata=metadata)
    core.read_namespaced_secret.return_value = client.V1Secret(metadata=metadata)
    account = client.V1ServiceAccount(
        metadata=client.V1ObjectMeta(resource_version="10"),
        image_pull_secrets=[
            client.V1LocalObjectReference(name="cluster-pull"),
            client.V1LocalObjectReference(name="e2e-registry-unit-run"),
        ],
    )
    core.read_namespaced_service_account.return_value = account

    def patch(name, namespace, body, **kwargs):
        if isinstance(body, list):
            index = int(body[-1]["path"].rsplit("/", 1)[1])
            assert account.image_pull_secrets[index].name == body[1]["value"]
            account.image_pull_secrets.pop(index)
        return account

    core.patch_namespaced_service_account.side_effect = patch

    def removed(*args, **kwargs):
        core.read_namespaced_secret.side_effect = ApiException(status=404)

    core.delete_namespaced_secret.side_effect = removed
    monkeypatch.setattr(registry.client, "CoreV1Api", Mock(return_value=core))
    return core


def test_secret_is_memory_only_and_attached_to_fresh_default_sa(authfile, secret_api, capsys):
    secret = registry.PullSecret("unit-ns", "unit-run")
    secret.create("namespace-uid")
    call = secret_api.create_namespaced_secret.call_args
    assert call.args[0] == "unit-ns"
    body = call.args[1]
    assert body.type == "kubernetes.io/dockerconfigjson"
    assert json.loads(body.string_data[".dockerconfigjson"]) == registry.quay_auth()
    assert secret.uid == "secret-uid"
    assert secret_api.patch_namespaced_service_account.call_args.args == (
        "default",
        "unit-ns",
        {"imagePullSecrets": [{"name": secret.name}]},
    )
    assert "synthetic-secret" not in capsys.readouterr().out


@pytest.mark.parametrize("change", ["namespace", "secret", "label", "owner"])
def test_cleanup_refuses_replaced_or_unowned_resources(secret_api, change):
    secret = registry.PullSecret("unit-ns", "unit-run")
    secret.namespace_uid, secret.uid, secret.attempted = "namespace-uid", "secret-uid", True
    if change == "namespace":
        secret_api.read_namespace.return_value.metadata.uid = "replacement"
    elif change == "secret":
        secret_api.read_namespaced_secret.return_value.metadata.uid = "replacement"
    elif change == "label":
        secret_api.read_namespaced_secret.return_value.metadata.labels = {}
    else:
        secret_api.read_namespaced_secret.return_value.metadata.owner_references = []
    with pytest.raises(RuntimeError, match="cleanup could not be confirmed"):
        secret.cleanup()
    secret_api.delete_namespaced_secret.assert_not_called()
    secret_api.patch_namespaced_service_account.assert_not_called()


@pytest.mark.parametrize("ambiguous_post", [False, True])
def test_secret_cleanup_uses_uid_and_detaches_only_own_reference(secret_api, ambiguous_post):
    secret = registry.PullSecret("unit-ns", "unit-run")
    secret.namespace_uid, secret.attempted = "namespace-uid", True
    secret.uid = None if ambiguous_post else "secret-uid"
    assert secret.cleanup() == "removed"
    assert (
        secret_api.delete_namespaced_secret.call_args.kwargs["body"].preconditions.uid
        == "secret-uid"
    )
    assert secret_api.patch_namespaced_service_account.call_args.args[2] == [
        {"op": "test", "path": "/metadata/resourceVersion", "value": "10"},
        {"op": "test", "path": "/imagePullSecrets/1/name", "value": secret.name},
        {"op": "remove", "path": "/imagePullSecrets/1"},
    ]
    assert [
        ref.name
        for ref in secret_api.read_namespaced_service_account.return_value.image_pull_secrets
    ] == ["cluster-pull"]


def test_secret_cleanup_absent_and_never_attempted(secret_api):
    secret = registry.PullSecret("unit-ns", "unit-run")
    assert secret.cleanup() == "not created"
    secret_api.read_namespace.assert_not_called()
    secret.namespace_uid, secret.attempted = "namespace-uid", True
    secret_api.read_namespaced_secret.side_effect = ApiException(status=404)
    assert secret.cleanup() == "already absent"
    secret_api.delete_namespaced_secret.assert_not_called()


def test_secret_creation_failure_is_redacted_and_remains_cleanable(authfile, secret_api):
    secret = registry.PullSecret("unit-ns", "unit-run")
    secret_api.create_namespaced_secret.side_effect = ApiException(
        status=504, reason="synthetic-secret"
    )
    with pytest.raises(RuntimeError) as error:
        secret.create("namespace-uid")
    assert secret.attempted and secret.uid is None
    assert "synthetic-secret" not in str(error.value)
    assert secret.cleanup() == "removed"


def test_private_preflight_is_opt_in(monkeypatch, tmp_path):
    args = deploy.arguments(
        ["--image-repository", "quay.io/test/service", "--values", str(tmp_path / "values.yaml")]
    )
    assert not args.use_podman_auth
    enabled = deploy.arguments(
        [
            "--image-repository",
            "quay.io/test/service",
            "--values",
            str(args.values),
            "--use-podman-auth",
        ]
    )
    assert enabled.use_podman_auth


def test_registry_preflight_is_readonly_and_checks_connection_before_auth(monkeypatch):
    api = MagicMock()
    monkeypatch.setattr(registry, "kubeconfig_client", Mock(return_value=api))
    version = Mock()
    monkeypatch.setattr(registry.client, "VersionApi", Mock(return_value=version))
    auth = Mock()
    monkeypatch.setattr(registry, "quay_auth", auth)
    version.get_code.side_effect = RuntimeError("synthetic-secret")
    with pytest.raises(ValueError, match="Cannot connect") as error:
        registry.preflight()
    auth.assert_not_called()
    assert "synthetic-secret" not in str(error.value)
    version.get_code.side_effect = None
    registry.preflight()
    auth.assert_called_once()


def test_default_sa_creation_race_is_bounded_and_retried(authfile, secret_api, monkeypatch):
    secret_api.patch_namespaced_service_account.side_effect = [ApiException(status=404), None]
    sleep = Mock()
    monkeypatch.setattr(registry.time, "sleep", sleep)
    secret = registry.PullSecret("unit-ns", "unit-run")
    secret.create("namespace-uid")
    sleep.assert_called_once_with(1)
    assert secret_api.patch_namespaced_service_account.call_count == 2


def test_secret_removed_even_if_sa_detachment_fails(secret_api):
    secret = registry.PullSecret("unit-ns", "unit-run")
    secret.namespace_uid, secret.uid, secret.attempted = "namespace-uid", "secret-uid", True
    secret_api.patch_namespaced_service_account.side_effect = ApiException(status=403)
    with pytest.raises(RuntimeError, match="cleanup could not be confirmed"):
        secret.cleanup()
    secret_api.delete_namespaced_secret.assert_called_once()


def test_accepted_delete_is_not_claimed_as_confirmed_cleanup(secret_api, monkeypatch):
    secret = registry.PullSecret("unit-ns", "unit-run")
    secret.namespace_uid, secret.uid, secret.attempted = "namespace-uid", "secret-uid", True
    secret_api.delete_namespaced_secret.side_effect = None
    monkeypatch.setattr(registry.time, "monotonic", Mock(side_effect=[0, 31]))
    with pytest.raises(RuntimeError, match="cleanup could not be confirmed"):
        secret.cleanup()


@pytest.mark.parametrize("conflicts", [1, 2])
def test_reference_cleanup_retries_concurrent_changes(secret_api, conflicts):
    patch = secret_api.patch_namespaced_service_account.side_effect
    calls = 0

    def conflict_then_remove(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= conflicts:
            raise ApiException(status=409)
        return patch(*args, **kwargs)

    secret_api.patch_namespaced_service_account.side_effect = conflict_then_remove
    registry.PullSecret("unit-ns", "unit-run").detach(secret_api)
    assert calls == conflicts + 1


def test_reference_cleanup_does_not_claim_success_after_noop_patch(secret_api):
    secret_api.patch_namespaced_service_account.side_effect = None
    with pytest.raises(RuntimeError, match="reference removal is not confirmed"):
        registry.PullSecret("unit-ns", "unit-run").detach(secret_api)
    assert secret_api.patch_namespaced_service_account.call_count == 3
