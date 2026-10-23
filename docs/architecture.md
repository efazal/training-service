# Server structure

```text
training-service/
├── docs/api/
│   ├── openapi.yaml                   # reviewed contract and dispatch metadata
│   ├── openapi-generator-config.yaml # python-fastapi settings
│   ├── format_generated.py           # value-preserving string wrapping
│   └── templates/python-fastapi/     # reproducible generator overrides
├── src/training_service_api/          # generated, committed; do not edit
├── src/training_service/
│   ├── main.py                       # generated routers, Swagger, probes
│   ├── api_impl/                     # handwritten submission handler
│   ├── job_submission.py             # caller-scoped CodeFlare adapter
│   ├── ray_entrypoint.py             # trusted code embedded in RayJob
│   ├── auth.py                       # TokenReview and request-scoped caller
│   ├── authorization.py              # project SubjectAccessReview
│   ├── contract.py                   # canonical model validation and Swagger
│   ├── contract_types.py             # scalar URI generator workaround
│   └── errors.py                     # API error envelope
├── tests/unit/                       # startup/routes, models, formatting
└── Containerfile                     # packages both Python namespaces
```

Only one FastAPI application runs. Generated routers and models form a
library imported by the handwritten entry point; the wheel and container
bundle both namespaces. Generation never overwrites handwritten source.

The application registers the contract's eight operations under `/api/v1`.
Authentication and project authorization run before submission. POST creates
a caller-scoped CodeFlare RayJob and returns 202. Other operations remain
fail-closed with `501`. `/healthz`, `/readyz`, `/docs`, and `/openapi.json`
are available without a cluster.

## Ownership boundary

Routes, request/response models, credential interfaces, and formatting are
generated. Canonical validation fills generator gaps without changing the
contract. Authentication, project authorization, CodeFlare/Ray calls, and Helm
deployment behavior are handwritten and remain outside generated code. The
POST-only generator extension passes request context to the handler and sets
the 202 status and Location header. CodeFlare builds the RayJob resource; the
submission adapter adds accepted non-secret request metadata before the
caller-authorized, SDK-owned create call. The caller-bound Kubernetes client
preserves RayJob create errors that CodeFlare's vendored transport otherwise
swallows, so the API can return the contract's 403/409/503 responses. The
service does not create a RayJob resource or author its spec. There is no
service-side job store.

The runtime Swagger copy changes only the server URL. The reviewed YAML
remains the source of truth.
