"""Explicit local Podman -> Quay -> Helm -> pytest workflow; never run by CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import yaml

from tests.e2e.support import PROFILES, namespace, workload

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "charts/training-service"


def command(args, *, timeout=120, capture=True, env=None):
    """Do not echo commands/output that could contain credentials or chart values."""
    try:
        result = subprocess.run(
            [str(arg) for arg in args],
            cwd=ROOT,
            env=env,
            timeout=timeout,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError(
            f"{args[0]} {args[1]} could not finish; check local tooling/session"
        ) from None
    if result.returncode:
        raise RuntimeError(
            f"{args[0]} {args[1]} failed (exit {result.returncode}); "
            "check credentials, permissions and connectivity. Raw output withheld."
        )
    return result.stdout or ""


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-repository", required=True, help="quay.io/your-org/training-service"
    )
    parser.add_argument(
        "--namespace", required=True, type=namespace, help="Existing dedicated namespace"
    )
    parser.add_argument("--values", required=True, type=Path, help="Credential-free Helm values")
    parser.add_argument("--platform", default="linux/amd64", choices=("linux/amd64", "linux/arm64"))
    parser.add_argument("--job-file", type=Path, help="Opt into real training/compute allocation")
    parser.add_argument("--queue-profile", choices=PROFILES)
    parser.add_argument("--artifact-pod", help="Prepared read-only PVC verifier pod in the project")
    parser.add_argument("--artifact-container", default="verifier")
    parser.add_argument("--forbidden-project", type=namespace)
    parser.add_argument(
        "--keep-deployment", action="store_true", help="Keep successful test deployment too"
    )
    parser.add_argument("--build-timeout", type=int, default=1800)
    parser.add_argument("--deploy-timeout", type=int, default=300)
    parser.add_argument("--job-timeout", type=int, default=1800)
    parser.add_argument("--stream-timeout", type=int, default=60)
    parser.add_argument("--cleanup-timeout", type=int, default=180)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "artifacts/e2e")
    return parser.parse_args(argv)


def preflight(args):
    if not re.fullmatch(r"quay\.io/[a-z0-9_-]+/[a-z0-9_.-]+", args.image_repository):
        raise ValueError("Use an untagged quay.io/<organization>/<repository> image repository")
    for option in (
        "build_timeout",
        "deploy_timeout",
        "job_timeout",
        "stream_timeout",
        "cleanup_timeout",
    ):
        if getattr(args, option) <= 0:
            raise ValueError(f"--{option.replace('_', '-')} must be positive")
    if not os.getenv("TRAINING_SERVICE_TOKEN", "").strip():
        raise ValueError("Set TRAINING_SERVICE_TOKEN in the environment (never a CLI argument)")
    for tool in ("podman", "helm", "oc", "git"):
        if not shutil.which(tool):
            raise ValueError(f"Required executable not found: {tool}")
    if not args.values.is_file():
        raise ValueError("--values must be a readable credential-free Helm values file")
    try:
        values = yaml.safe_load(args.values.read_text())
    except yaml.YAMLError:
        raise ValueError("--values must be valid YAML; raw values are not logged") from None
    if not isinstance(values, dict):
        raise ValueError("Helm values must contain an object")
    if args.queue_profile and not args.job_file:
        raise ValueError("--queue-profile requires --job-file")
    if args.artifact_pod and not args.job_file:
        raise ValueError("--artifact-pod requires --job-file")
    if args.job_file:
        payload = workload(args.job_file, args.queue_profile)
        # Validate observer/profile prerequisites BEFORE building or deploying.
        from tests.e2e.observer import Observer

        try:
            with Observer(args.namespace) as observer:
                observer.preflight(payload, args.queue_profile)
                if args.artifact_pod:
                    observer.artifact_mount(args.artifact_pod, args.artifact_container, payload)
        except Exception:
            raise ValueError(
                "Observer preflight failed: check kubeconfig, profile and permissions"
            ) from None
    ssl.create_default_context(cafile=os.getenv("TRAINING_SERVICE_CA_BUNDLE") or None)
    command(["podman", "info", "--format", "{{.Host.Arch}}"])
    command(["podman", "login", "--get-login", "quay.io"])
    command(["oc", "get", "namespace", args.namespace, "-o", "name"])
    nodes = json.loads(command(["oc", "get", "nodes", "-o", "json"]))["items"]
    architecture = args.platform.partition("/")[2]
    if not any(n["metadata"]["labels"].get("kubernetes.io/arch") == architecture for n in nodes):
        raise ValueError("--platform has no matching cluster node architecture")
    for resource in (
        "deployments",
        "services",
        "routes.route.openshift.io",
        "secrets",
        "serviceaccounts",
    ):
        allowed = command(["oc", "auth", "can-i", "create", resource, "-n", args.namespace]).strip()
        if allowed != "yes":
            raise ValueError(f"Deployment identity needs create access to {resource}")
    command(["helm", "lint", CHART, "-f", args.values])


def wait_ready(url, timeout):
    tls = ssl.create_default_context(cafile=os.getenv("TRAINING_SERVICE_CA_BUNDLE") or None)
    deadline = time.monotonic() + timeout
    with httpx.Client(verify=tls, follow_redirects=False, trust_env=False, timeout=10) as http:
        while time.monotonic() < deadline:
            try:
                if http.get(url + "/readyz").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
    raise RuntimeError("Route readiness/TLS deadline exceeded; check ingress CA and deployment")


def run(args):
    preflight(args)
    commit = command(["git", "rev-parse", "HEAD"]).strip()
    run_id = uuid.uuid4().hex[:10]
    release = "training-e2e-" + run_id
    tag = "e2e-" + commit[:12] + "-" + run_id
    image = args.image_repository + ":" + tag
    results = args.results_dir.resolve() / run_id
    results.mkdir(parents=True, exist_ok=False)
    evidence = {
        "commit": commit,
        "dirty": bool(command(["git", "status", "--porcelain"]).strip()),
        "namespace": args.namespace,
        "release": release,
        "profile": args.queue_profile or "smoke",
        "outcome": "incomplete",
        "chart_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(CHART.rglob("*")) if p.is_file())
        ).hexdigest(),
    }
    deployed = False
    succeeded = False
    try:
        print(f"Building {image} for {args.platform}", flush=True)
        command(
            [
                "podman",
                "build",
                "--platform",
                args.platform,
                "-f",
                "Containerfile",
                "-t",
                image,
                ".",
            ],
            timeout=args.build_timeout,
        )
        print("Pushing to Quay (using existing Podman login)", flush=True)
        digest_file = results / "image-digest.txt"
        command(["podman", "push", "--digestfile", digest_file, image], timeout=args.build_timeout)
        digest = digest_file.read_text().strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise RuntimeError("Podman did not return a valid pushed image digest")
        evidence["image"] = args.image_repository + "@" + digest
        print(f"Deploying isolated release {release} in {args.namespace}", flush=True)
        existing = json.loads(
            command(
                [
                    "helm",
                    "list",
                    "-n",
                    args.namespace,
                    "--all",
                    "--filter",
                    "^" + release + "$",
                    "-o",
                    "json",
                ]
            )
        )
        if existing:
            raise RuntimeError("Generated release name already exists; refusing to adopt it")
        # Names are unique: never upgrade/adopt an existing release or namespace.
        deployed = True  # Includes partially failed installations for diagnostics.
        command(
            [
                "helm",
                "install",
                release,
                CHART,
                "-n",
                args.namespace,
                "-f",
                args.values,
                "--set-string",
                "fullnameOverride=" + release,
                "--set-string",
                "serviceAccount.name=" + release,
                "--set",
                "serviceAccount.create=true",
                "--set-string",
                "image.repository=" + args.image_repository,
                "--set-string",
                "image.tag=" + tag + "@" + digest,
                "--set",
                "image.pullPolicy=Always",
                "--set",
                "route.enabled=true",
                "--set",
                "route.tls.enabled=true",
                "--set",
                "route.path=",
                "--set",
                "route.host=",
                "--wait",
                "--timeout",
                f"{args.deploy_timeout}s",
            ],
            timeout=args.deploy_timeout + 30,
        )
        route = json.loads(
            command(["oc", "get", "route", release, "-n", args.namespace, "-o", "json"])
        )
        host = route["spec"]["host"]
        if not re.fullmatch(r"[a-zA-Z0-9.-]+", host):
            raise RuntimeError("Route returned an invalid hostname")
        url = "https://" + host
        evidence["url"] = url
        wait_ready(url, args.deploy_timeout)
        pods = json.loads(
            command(
                [
                    "oc",
                    "get",
                    "pods",
                    "-n",
                    args.namespace,
                    "-l",
                    "app.kubernetes.io/instance=" + release,
                    "-o",
                    "json",
                ]
            )
        )
        evidence["running_image_ids"] = [
            c.get("imageID")
            for p in pods["items"]
            for c in p.get("status", {}).get("containerStatuses", [])
        ]
        versions = json.loads(
            command(
                [
                    "oc",
                    "exec",
                    "-n",
                    args.namespace,
                    "deployment/" + release,
                    "--",
                    "python",
                    "-c",
                    "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) "
                    "for k in ['training-service','ray','codeflare-sdk']}))",
                ]
            )
        )
        evidence["service_component_versions"] = {
            k: versions[k] for k in ("training-service", "ray", "codeflare-sdk")
        }
        values = yaml.safe_load(args.values.read_text())
        evidence["runtime_image"] = next(
            (
                e.get("value", "valueFrom: not inspected")
                for e in values.get("env", [])
                if e.get("name") == "TRAINING_RAY_IMAGE"
            ),
            "not observed: configured through envFrom",
        )
        evidence["runtime_training_hub_version"] = (
            "not observed; record tested runtime release separately"
        )
        evidence["cluster_versions"] = json.loads(command(["oc", "version", "-o", "json"]))
        env = dict(
            os.environ,
            TRAINING_SERVICE_URL=url,
            TRAINING_SERVICE_PROJECT=args.namespace,
            TRAINING_SERVICE_EVIDENCE_DIR=str(results),
        )
        test_args = [
            sys.executable,
            "-m",
            "pytest",
            "tests/e2e",
            "-s",
            "--tb=short",
            "--junitxml",
            str(results / "junit.xml"),
            "--job-timeout",
            str(args.job_timeout),
            "--stream-timeout",
            str(args.stream_timeout),
            "--cleanup-timeout",
            str(args.cleanup_timeout),
        ]
        if args.job_file:
            test_args += [
                "--job-file",
                str(args.job_file),
                "--queue-profile",
                args.queue_profile,
                "--observe-cluster",
                "--check-logs",
                "--check-stream",
            ]
        if args.artifact_pod:
            test_args += [
                "--artifact-pod",
                args.artifact_pod,
                "--artifact-container",
                args.artifact_container,
            ]
        if args.forbidden_project:
            test_args += ["--forbidden-project", args.forbidden_project]
        print(f"Testing {url}; results: {results}", flush=True)
        # Live tests produce only sanitized assertions; no --showlocals or verbose HTTP logging.
        command(
            test_args, timeout=args.job_timeout + args.cleanup_timeout + 600, capture=False, env=env
        )
        succeeded = True
        evidence["outcome"] = "passed"
    finally:
        try:
            if deployed and succeeded and not args.keep_deployment:
                try:
                    command(
                        [
                            "helm",
                            "uninstall",
                            release,
                            "-n",
                            args.namespace,
                            "--wait",
                            "--timeout",
                            f"{args.cleanup_timeout}s",
                        ],
                        timeout=args.cleanup_timeout + 30,
                    )
                    evidence["deployment_cleanup"] = "removed"
                except RuntimeError:
                    evidence["outcome"] = "deployment_cleanup_failed"
                    evidence["deployment_cleanup"] = "incomplete"
                    print(f"Cleanup incomplete: helm uninstall {release} -n {args.namespace}")
                    raise
            elif deployed:
                evidence["deployment_cleanup"] = "retained"
                print(f"Retained deployment. Cleanup: helm uninstall {release} -n {args.namespace}")
        finally:
            (results / "run.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(f"Evidence: {results}", flush=True)
            print("Quay image and training outputs are retained; no customer storage was removed.")


def main():
    try:
        run(arguments())
    except (ValueError, RuntimeError) as error:
        print(f"E2E workflow failed: {error}", file=sys.stderr)
        return 1
    except (OSError, ssl.SSLError):
        print(
            "E2E workflow failed; see the last stage above. Check documented prerequisites, "
            "login/TLS, and retained resources. No ambiguous POST is retried.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted: inspect the reported job/deployment; active jobs are not force-deleted.",
            file=sys.stderr,
        )
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
