"""Load the canonical contract and validate generated component models."""

from copy import deepcopy
from functools import cache, lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft4Validator, ValidationError
from pydantic import BaseModel, model_validator


@lru_cache(maxsize=1)
def load_contract() -> dict[str, Any]:
    """Use the packaged spec, or the source-tree spec in editable development."""
    packaged = files("training_service").joinpath("openapi.yaml")
    if packaged.is_file():
        text = packaged.read_text(encoding="utf-8")
    else:
        text = (Path(__file__).resolve().parents[2] / "docs" / "api" / "openapi.yaml").read_text(
            encoding="utf-8"
        )
    result: dict[str, Any] = yaml.safe_load(text)
    return result


def contract_document() -> dict[str, Any]:
    """Preserve reviewed schemas/examples while targeting this deployment."""
    document = deepcopy(load_contract())
    document["servers"] = [{"url": "/api/v1"}]
    for path in document["paths"].values():
        for operation in path.values():
            if isinstance(operation, dict) and operation.get("operationId") in {
                "estimateTrainingJob",
                "listAlgorithms",
                "listQueues",
            }:
                operation["responses"]["501"] = {
                    "description": "Not implemented in this local scaffold.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    },
                }
    return document


def json_schema(value: Any) -> Any:
    """Convert OpenAPI 3.0 nullable types without changing the source contract."""
    if isinstance(value, list):
        return [json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: json_schema(item) for key, item in value.items() if key != "nullable"}
    if value.get("nullable") and "type" in result:
        result["type"] = [result["type"], "null"]
    return result


@cache
def component_validator(name: str) -> Draft4Validator:
    return Draft4Validator(
        {
            "components": json_schema(load_contract()["components"]),
            "$ref": "#/components/schemas/" + name,
        }
    )


def json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_unset=True)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


class ContractModel(BaseModel):
    """Close generator validation gaps without rewriting the OpenAPI schema."""

    @model_validator(mode="before")
    @classmethod
    def validate_component(cls, value: Any) -> Any:
        if isinstance(value, dict) and cls.__name__ in load_contract()["components"]["schemas"]:
            try:
                component_validator(cls.__name__).validate(json_value(value))
            except ValidationError as error:
                raise ValueError(f"Payload does not match {cls.__name__} contract") from error
        return value
