"""Opt-in, namespace-local registry credentials; never write them to evidence."""

import base64
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

RUN_LABEL = "training.rhoai.io/e2e-run"
TIMEOUT = 15


def kubeconfig_client():
    """Honor the caller's kubeconfig without enabling credential debug logging."""
    settings = client.Configuration()
    config.load_kube_config(client_configuration=settings)
    if settings.debug or logging.getLogger("kubernetes.client.rest").isEnabledFor(logging.DEBUG):
        raise ValueError(
            "Private registry support requires Kubernetes HTTP debug logging to be disabled "
            "so registry credentials are not logged."
        )
    return client.ApiClient(settings)


def auth_paths():
    """Follow Podman's auth-file locations; an explicit override is authoritative."""
    if override := os.getenv("REGISTRY_AUTH_FILE"):
        return [Path(override)]
    paths = []
    if sys.platform == "linux":
        runtime = Path(os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        paths.append(runtime / "containers/auth.json")
    configuration = Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    paths.append(configuration / "containers/auth.json")
    paths.append(Path(os.getenv("DOCKER_CONFIG", str(Path.home() / ".docker"))) / "config.json")
    return paths


def quay_auth():
    """Return only Quay basic auth, resolving configured credential helpers in memory."""
    try:
        for path in auth_paths():
            if not path.is_file():
                continue
            document = json.loads(path.read_text())
            helper = document.get("credHelpers", {}).get("quay.io") or document.get("credsStore")
            if helper:
                if not isinstance(helper, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", helper):
                    raise ValueError
                result = subprocess.run(
                    ["docker-credential-" + helper, "get"],
                    input="quay.io\n",
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=TIMEOUT,
                )
                if result.returncode:
                    raise ValueError
                credential = json.loads(result.stdout)
                username, password = credential["Username"], credential["Secret"]
                if not isinstance(username, str) or not isinstance(password, str):
                    raise ValueError
                encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
            else:
                entry = document.get("auths", {}).get("quay.io", {})
                encoded = entry.get("auth")
                if not encoded:
                    continue
                username, separator, password = (
                    base64.b64decode(encoded, validate=True).decode().partition(":")
                )
                if not separator:
                    raise ValueError
            if not username or not password or ":" in username:
                raise ValueError
            return {"auths": {"quay.io": {"auth": encoded}}}
    except (OSError, ValueError, TypeError, AttributeError, KeyError, subprocess.SubprocessError):
        raise ValueError(
            "Cannot resolve Quay credentials safely. Run podman login quay.io; "
            "use REGISTRY_AUTH_FILE for a nondefault Podman auth file. Raw credentials withheld."
        ) from None
    raise ValueError(
        "No Quay credentials found. Run podman login quay.io, or set REGISTRY_AUTH_FILE "
        "to the auth file used for that login."
    )


def preflight():
    """Check the configured connection before reading credentials; no cluster writes."""
    try:
        with kubeconfig_client() as api:
            if not api.configuration.verify_ssl:
                print(
                    "WARNING: kubeconfig disables Kubernetes API certificate verification. "
                    "Honoring that existing setting; Quay credentials will be sent without "
                    "verifying API-server identity. Use only with a trusted development cluster.",
                    flush=True,
                )
            client.VersionApi(api).get_code(_request_timeout=TIMEOUT)
    except ValueError:
        raise
    except Exception:
        raise ValueError(
            "Cannot connect to the Kubernetes API for private registry support; "
            "check kubeconfig, permissions and connectivity. Raw error withheld."
        ) from None
    quay_auth()


class PullSecret:
    def __init__(self, namespace, run_id):
        self.namespace, self.run_id = namespace, run_id
        self.name = "e2e-registry-" + run_id
        self.namespace_uid = None
        self.uid = None
        self.attempted = False

    def check_namespace(self, core):
        current = core.read_namespace(self.namespace, _request_timeout=TIMEOUT)
        if (
            current.metadata.uid != self.namespace_uid
            or (current.metadata.labels or {}).get(RUN_LABEL) != self.run_id
        ):
            raise RuntimeError(
                "Namespace ownership changed; refusing registry credential operation"
            )

    def create(self, namespace_uid):
        self.namespace_uid = namespace_uid
        try:
            with kubeconfig_client() as api:
                core = client.CoreV1Api(api)
                self.check_namespace(core)
                body = client.V1Secret(
                    metadata=client.V1ObjectMeta(
                        name=self.name,
                        namespace=self.namespace,
                        labels={RUN_LABEL: self.run_id},
                        owner_references=[
                            client.V1OwnerReference(
                                api_version="v1",
                                kind="Namespace",
                                name=self.namespace,
                                uid=namespace_uid,
                            )
                        ],
                    ),
                    type="kubernetes.io/dockerconfigjson",
                    string_data={".dockerconfigjson": json.dumps(quay_auth())},
                )
                self.attempted = True  # A timed-out POST may still create the Secret.
                created = core.create_namespaced_secret(
                    self.namespace, body, _request_timeout=TIMEOUT
                )
                self.uid = created.metadata.uid
                # Ray's default-SA pods (including its submitter) inherit pull credentials.
                # This SA is in our new namespace; existing namespace policy is never changed.
                deadline = time.monotonic() + 30
                while True:
                    try:
                        account = core.read_namespaced_service_account(
                            "default", self.namespace, _request_timeout=TIMEOUT
                        )
                        if any(ref.name == self.name for ref in account.image_pull_secrets or []):
                            break
                        # ServiceAccount pull secrets are atomic; append without replacing
                        # controller-injected references, and retry if the account changes.
                        core.patch_namespaced_service_account(
                            "default",
                            self.namespace,
                            [
                                {
                                    "op": "test",
                                    "path": "/metadata/resourceVersion",
                                    "value": account.metadata.resource_version,
                                },
                                {
                                    "op": "add",
                                    "path": (
                                        "/imagePullSecrets/-"
                                        if account.image_pull_secrets
                                        else "/imagePullSecrets"
                                    ),
                                    "value": (
                                        {"name": self.name}
                                        if account.image_pull_secrets
                                        else [{"name": self.name}]
                                    ),
                                },
                            ],
                            _request_timeout=TIMEOUT,
                        )
                        break
                    except ApiException as error:
                        if error.status not in {404, 409, 422} or time.monotonic() >= deadline:
                            raise
                        time.sleep(1)  # Wait for default-SA creation or a concurrent update.
        except Exception:
            raise RuntimeError(
                "Cannot provision the test-only Quay pull secret; check kubeconfig, "
                "Secret/serviceaccount permissions and Podman login. Raw error withheld."
            ) from None

    def detach(self, core):
        """Remove only our SA reference, guarding concurrent controller updates."""
        # Three patch attempts plus a final read to confirm the last successful patch.
        for attempt in range(4):
            try:
                account = core.read_namespaced_service_account(
                    "default",
                    self.namespace,
                    _request_timeout=TIMEOUT,
                )
            except ApiException as error:
                if error.status == 404:
                    return
                raise
            index = next(
                (
                    i
                    for i, ref in enumerate(account.image_pull_secrets or [])
                    if ref.name == self.name
                ),
                None,
            )
            if index is None:
                return
            if attempt == 3:
                break
            path = f"/imagePullSecrets/{index}"
            try:
                core.patch_namespaced_service_account(
                    "default",
                    self.namespace,
                    [
                        {
                            "op": "test",
                            "path": "/metadata/resourceVersion",
                            "value": account.metadata.resource_version,
                        },
                        {"op": "test", "path": path + "/name", "value": self.name},
                        {"op": "remove", "path": path},
                    ],
                    _request_timeout=TIMEOUT,
                )
            except ApiException as error:
                if error.status not in {409, 422}:
                    raise
        raise RuntimeError("Registry pull-secret reference removal is not confirmed")

    def cleanup(self):
        if not self.attempted:
            return "not created"
        try:
            with kubeconfig_client() as api:
                core = client.CoreV1Api(api)
                self.check_namespace(core)
                try:
                    secret = core.read_namespaced_secret(
                        self.name, self.namespace, _request_timeout=TIMEOUT
                    )
                except ApiException as error:
                    if error.status != 404:
                        raise
                    self.detach(core)
                    return "already absent"
                if (
                    (secret.metadata.labels or {}).get(RUN_LABEL) != self.run_id
                    or (self.uid and secret.metadata.uid != self.uid)
                    or not any(
                        ref.uid == self.namespace_uid and ref.kind == "Namespace"
                        for ref in secret.metadata.owner_references or []
                    )
                ):
                    raise RuntimeError("Registry Secret ownership changed")
                core.delete_namespaced_secret(
                    self.name,
                    self.namespace,
                    body=client.V1DeleteOptions(
                        preconditions=client.V1Preconditions(uid=secret.metadata.uid)
                    ),
                    _request_timeout=TIMEOUT,
                )
                self.detach(core)
                deadline = time.monotonic() + 30
                while True:
                    try:
                        remaining = core.read_namespaced_secret(
                            self.name,
                            self.namespace,
                            _request_timeout=TIMEOUT,
                        )
                    except ApiException as error:
                        if error.status == 404:
                            return "removed"
                        raise
                    if (
                        remaining.metadata.uid != secret.metadata.uid
                        or time.monotonic() >= deadline
                    ):
                        raise RuntimeError("Registry Secret deletion is not confirmed")
                    time.sleep(1)
        except Exception:
            raise RuntimeError(
                f"Pull-secret cleanup could not be confirmed: inspect Secret '{self.name}' "
                f"in namespace '{self.namespace}'. No unrelated Secret is deleted."
            ) from None
