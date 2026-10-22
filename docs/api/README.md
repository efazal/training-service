# API generation

`openapi.yaml` is the reviewed public contract (version `1.4.0-dp`). The
contract is copied unchanged from the reviewed Training API specification;
generation does not modify its endpoints or schemas.

Generate with a running Podman or Docker engine:

```bash
make generate-server
```

The image is pinned to `openapitools/openapi-generator-cli:v7.25.0` and uses
the `python-fastapi` generator. Output goes directly to the committed
`src/training_service_api/` package. Its README and every Python file identify
it as generated and not to be edited directly. The wheel bundles that
package and the canonical YAML. Regenerate before installing dependencies or
building the container. `make install` runs generation automatically.

Generation emits only the API library: routers, base interfaces, models,
credential interfaces, and package markers. It does not create a second
application, Containerfile, dependency manifest, or test suite.

The overrides in `templates/python-fastapi/` are based on upstream v7.25.0
templates. They preserve Pydantic serialization and extra-field rules,
forward the credential interface to future implementations, and import the
external scalar URI type. `ContractModel` supplies canonical schema
validation where the beta generator drops constraints. The external URI
mapping produces upstream lookup warnings; the template supplies its import,
and focused unit tests verify request examples round-trip unchanged.

Authentication uses OpenShift TokenReview; project operations also perform
SubjectAccessReview before handwritten implementations are invoked.
Job listing, detail, logs, and terminal deletion are implemented outside the
generated tree. Other operations return `501`. Operations marked
`x-request-context` receive the authenticated request explicitly. The query
argument override permits normal integer/boolean HTTP query parsing without
weakening strict JSON body models.

Swagger's runtime copy uses a relative server URL and documents the
scaffold's `501` response. It never overwrites the canonical YAML.

## Formatting and regeneration

`make generate-server` wraps long description literals, normalizes imports,
and runs Ruff formatting/linting. An AST comparison ensures wrapping does
not change string values or program semantics. Model/enum docstrings refer
to the canonical schema instead of repeating release plans.

Both handwritten and generated Python use the 100-column Ruff configuration.
Use `make format` for local formatting and `make check` for verification.
Changes live in templates and tooling, so regeneration preserves them and
leaves handwritten source untouched.

Commit the contract/templates and regenerated sources together. CI regenerates
the package and checks that the committed output is current. Generator
bookkeeping is ignored; API source is not.
