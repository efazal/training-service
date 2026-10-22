# Server scaffold structure

```text
training-service/
├── docs/api/
│   ├── openapi.yaml                   # canonical public contract
│   ├── openapi-generator-config.yaml # python-fastapi settings
│   ├── format_generated.py           # value-preserving string wrapping
│   └── templates/python-fastapi/     # reproducible generator overrides
├── src/training_service_api/          # generated, committed; do not edit
├── src/training_service/
│   ├── main.py                       # generated routers, Swagger, probes
│   ├── api_impl/                     # job reads/deletion and driver logs
│   ├── auth.py                       # OpenShift TokenReview
│   ├── authorization.py              # project SubjectAccessReview
│   ├── kube.py                       # isolated caller-token clients
│   ├── jobs.py                       # stateless RayJob views and deletion
│   ├── ray_logs.py                   # authenticated Ray SDK log access
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
Authentication and project authorization run before job operations. Reads
remain available from caller-authorized RayJob resources after RayCluster
cleanup. Driver logs use the Ray SDK through a trusted, authenticated gateway.
Terminal deletion uses foreground Kubernetes garbage collection, guarded by
ownership and resource-version checks. There is no service job database or
elevated workload identity. See [job operations](job-operations.md).
Unimplemented operations return `501`; probes and Swagger remain available.

## Ownership boundary

Routes, request/response models, credential interfaces, and formatting are
generated. Canonical validation fills generator gaps without changing the
contract. Authentication, project authorization, Kubernetes discovery,
CodeFlare/Ray calls, and Helm deployment behavior belong to separate tasks
and remain outside generated code.

The runtime Swagger copy changes only the server URL and documents the
scaffold's `501` responses. The reviewed YAML remains the source of truth.
