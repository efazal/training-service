"""Read OpenShift platform settings and print a Helm values override (no cluster writes)."""

import argparse
import json
import re
import ssl
import subprocess
import sys

import yaml


def read_json(arguments):
    result = subprocess.run(
        ["oc", "get", *arguments, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "Cannot read platform configuration. Check oc login and get permissions for "
            "GatewayConfig and the public ingress CA ConfigMap. "
            "Alternatively, configure logBackend.gatewayURL and optional trustedCA manually."
        )
    return json.loads(result.stdout)


def platform_values(read=read_json, *, system_trust=False):
    gateway = read(["gatewayconfigs.services.platform.opendatahub.io", "default-gateway"])
    domain = gateway.get("status", {}).get("domain", "")
    if not isinstance(domain, str) or not re.fullmatch(
        r"[A-Za-z0-9](?:[-A-Za-z0-9.]*[A-Za-z0-9])?", domain
    ):
        raise ValueError(
            "The platform GatewayConfig has no valid status.domain; wait for readiness."
        )
    settings = {"gatewayURL": "https://" + domain}
    if not system_trust:
        configmap = read(["configmap", "default-ingress-cert", "-n", "openshift-config-managed"])
        bundle = configmap.get("data", {}).get("ca-bundle.crt", "")
        if not isinstance(bundle, str) or not bundle or "PRIVATE KEY" in bundle:
            raise ValueError(
                "The platform ingress CA ConfigMap must contain public PEM certificates."
            )
        try:
            ssl.create_default_context().load_verify_locations(cadata=bundle)
        except (ValueError, ssl.SSLError):
            raise ValueError(
                "The platform ingress CA ConfigMap has an invalid certificate bundle."
            ) from None
        settings["trustedCA"] = {"bundle": bundle}
    return {"logBackend": settings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-trust",
        action="store_true",
        help="Omit the ingress CA bundle when the gateway uses the container's normal trust.",
    )
    args = parser.parse_args()
    try:
        values = platform_values(system_trust=args.system_trust)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Platform configuration failed: {error}\n")
    yaml.safe_dump(values, sys.stdout, sort_keys=False)


if __name__ == "__main__":
    main()
