SHELL := /bin/bash

UV ?= uv
OPENAPI_GENERATOR_VERSION ?= 7.25.0
CONTAINER_RUNTIME ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
IMAGE_PLATFORM ?=
IMAGE_REPOSITORY ?= quay.io/modh/training-service
IMAGE_TAG ?= 0.1.0
IMAGE ?= $(IMAGE_REPOSITORY):$(IMAGE_TAG)
IMAGE_SMOKE_PORT ?= 18080
IMAGE_SMOKE_CONTAINER ?= training-service-smoke
CURL ?= curl
HELM_RELEASE ?= training-service
HELM_NAMESPACE ?= training-service
HELM_CHART ?= charts/training-service
HELM_OUTPUT_DIR ?= dist
CHART_VERSION ?= 0.1.0
APP_VERSION ?= $(CHART_VERSION)
E2E_ARGS ?=

.PHONY: install test test-e2e test-e2e-deploy lint format format-check typecheck check run generate-server \
	image-build image-push image-smoke helm-check helm-lint helm-template helm-package

install: generate-server
	$(UV) sync

test:
	$(UV) run pytest

# Existing deployment, or Podman -> Quay -> Helm -> live HTTP tests.
test-e2e:
	$(UV) run --locked pytest tests/e2e $(E2E_ARGS)

test-e2e-deploy:
	$(UV) run --locked python -m tests.e2e.deploy $(E2E_ARGS)

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff check --fix --select I .
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy src

check: lint format-check typecheck test

run:
	$(UV) run training-service

image-build:
	@test -n "$(CONTAINER_RUNTIME)" || (echo "podman or docker is required" >&2; exit 1)
	$(CONTAINER_RUNTIME) build $(if $(IMAGE_PLATFORM),--platform $(IMAGE_PLATFORM),) \
		-f Containerfile -t $(IMAGE) .

image-push:
	@test -n "$(CONTAINER_RUNTIME)" || (echo "podman or docker is required" >&2; exit 1)
	$(CONTAINER_RUNTIME) push $(IMAGE)

# Starts the API image without a runtime catalog: health must pass, readiness
# must fail closed. Helm supplies the catalog in a real deployment.
image-smoke:
	@test -n "$(CONTAINER_RUNTIME)" || (echo "podman or docker is required" >&2; exit 1)
	@command -v $(CURL) >/dev/null || (echo "curl is required" >&2; exit 1)
	@set -e; \
	$(CONTAINER_RUNTIME) rm -f $(IMAGE_SMOKE_CONTAINER) >/dev/null 2>&1 || true; \
	trap '$(CONTAINER_RUNTIME) rm -f $(IMAGE_SMOKE_CONTAINER) >/dev/null 2>&1 || true' EXIT; \
	$(CONTAINER_RUNTIME) run --detach --name $(IMAGE_SMOKE_CONTAINER) \
		--publish $(IMAGE_SMOKE_PORT):8080 $(IMAGE); \
	for attempt in $$(seq 1 30); do \
		if $(CURL) --fail --silent --max-time 2 http://127.0.0.1:$(IMAGE_SMOKE_PORT)/healthz >/dev/null \
			&& test "$$($(CURL) --silent --output /dev/null --write-out '%{http_code}' --max-time 2 http://127.0.0.1:$(IMAGE_SMOKE_PORT)/readyz)" = 503; then \
			echo "image smoke check passed"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	$(CONTAINER_RUNTIME) logs $(IMAGE_SMOKE_CONTAINER); \
	echo "image smoke check failed" >&2; \
	exit 1

generate-server:
	@test -n "$(CONTAINER_RUNTIME)" || (echo "podman or docker is required" >&2; exit 1)
	$(CONTAINER_RUNTIME) run --rm -v "$$(pwd):/local" \
		openapitools/openapi-generator-cli:v$(OPENAPI_GENERATOR_VERSION) generate \
		-i /local/docs/api/openapi.yaml \
		-g python-fastapi \
		-o /local \
		-c /local/docs/api/openapi-generator-config.yaml \
		-t /local/docs/api/templates/python-fastapi \
		--schema-mappings TrainingInputURI=TrainingInputURI \
		--import-mappings 'TrainingInputURI=from training_service.contract_types import TrainingInputURI' \
		--global-property 'apis,models,supportingFiles=security_api.py:extra_models.py:__init__.py,apiTests=false,modelTests=false,apiDocs=false,modelDocs=false'
	$(UV) run --frozen python docs/api/format_generated.py src/training_service_api
	$(UV) run --frozen ruff check --fix \
		--select I,F401,UP src/training_service_api
	$(UV) run --frozen ruff format src/training_service_api
	$(UV) run --frozen ruff check src/training_service_api


helm-check:
	@command -v helm >/dev/null || (echo "helm is required" >&2; exit 1)

helm-lint: helm-check
	helm lint $(HELM_CHART)

helm-template: helm-check
	helm template $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(HELM_NAMESPACE) >/dev/null
	helm template $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(HELM_NAMESPACE) \
		--set route.enabled=true \
		--set route.host=training-service.example.com \
		--set route.tls.enabled=true >/dev/null

helm-package: helm-check
	mkdir -p $(HELM_OUTPUT_DIR)
	helm package $(HELM_CHART) \
		--destination $(HELM_OUTPUT_DIR) \
		--version $(CHART_VERSION) \
		--app-version $(APP_VERSION)
