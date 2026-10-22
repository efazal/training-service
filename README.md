# OpenDataHub Training Service

Python/FastAPI scaffold generated from the reviewed Training API contract in
`docs/api/openapi.yaml`. The runnable application registers all eight contract
operations and serves Swagger UI at `/docs`.

OpenShift authentication and project authorization protect API operations.
Job listing, details, driver logs, and terminal RayJob deletion are implemented.
Submission, algorithm/queue discovery, and estimation remain separate work and
return `501 Not Implemented`. Health and readiness probes remain available.

See [job operations](docs/job-operations.md) for permissions, log gateway
configuration, response examples, and cascading cleanup behavior.

## Local development

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a running
Podman or Docker engine for OpenAPI Generator.

```bash
make install       # generate server, then install dependencies
make check         # lint, formatting, typing, focused unit tests
make run
```

Open http://localhost:8080/docs. Probes are at `/healthz` and `/readyz`.

Generated code lives in the committed `src/training_service_api/` package.
Its README and every Python file identify it as generated and not to be edited
directly. Edit the contract or templates, run `make generate-server`, and
commit the updated sources. Handwritten behavior belongs in
`src/training_service/api_impl/` and supporting modules.

Generate before building the existing container:

```bash
make generate-server
podman build -f Containerfile -t training-service:dev .
```

The Python wheel and container include the generated package and canonical
contract. Helm/deployment changes and backend implementations are separate
tasks.

See [API generation](docs/api/README.md) and [structure](docs/architecture.md).
