# API generation

`openapi.yaml` is the reviewed public contract (version `1.4.0-dp`).
Generation does not modify its endpoints or schemas. The POST-only
`x-submission-context` vendor extension controls generated dispatch and
response handling without changing the public wire schema.

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
forward the verified request context to the submission handler, and import
the external scalar URI type. `ContractModel` supplies canonical schema
validation where the beta generator drops constraints. The external URI
mapping produces upstream lookup warnings; the template supplies its import,
and focused unit tests verify request examples round-trip unchanged.

Authentication, project authorization, and POST submission are handwritten
outside the generated tree. The other operations remain fail-closed with
`501` after authentication and authorization.

Swagger's runtime copy uses a relative server URL and never overwrites the
canonical YAML.

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
