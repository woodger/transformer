from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Literal, cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import Schema, SchemaRegistry

from app.contracts.json_types import JsonObject

FlightRequestSchema = Literal[
    "acquire",
    "cancel",
    "create",
    "input-close",
    "inputs-list",
    "model-describe",
    "outputs-list",
    "query",
    "status",
    "upload-metadata",
]


class FlightContractError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        validation_error: ValidationError | None = None,
    ) -> None:
        super().__init__(message)
        self.validation_error = validation_error


_SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
_CONTRACTS_DIRECTORY = Path(__file__).parents[2]
_SCHEMA_DIRECTORIES = (
    _SCHEMA_DIRECTORY,
    _CONTRACTS_DIRECTORY / "semantic" / "v1" / "schemas",
)
_SCHEMA_NAMES = frozenset({
    "acquire",
    "cancel",
    "create",
    "input-close",
    "inputs-list",
    "model-describe",
    "outputs-list",
    "query",
    "status",
    "upload-metadata",
    "job-config",
    "capabilities-result",
    "health-result",
    "create-result",
    "acquire-result",
    "status-result",
    "inputs-list-result",
    "input-close-result",
    "outputs-list-result",
    "cancel-result",
    "model-describe-result",
    "put-result",
})


def validate_request_document(
    document: object,
    schema_name: str,
) -> JsonObject:
    if schema_name not in _SCHEMA_NAMES:
        raise ValueError(f"unknown Flight contract schema: {schema_name}")
    if not isinstance(document, dict):
        raise FlightContractError("Flight request must be a JSON object")
    typed_document = cast(JsonObject, document)
    errors = sorted(
        _validator(schema_name).iter_errors(typed_document),
        key=lambda error: (
            int(error.validator in {"allOf", "anyOf", "not", "oneOf"}),
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
        ),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{location}: " if location else ""
        raise FlightContractError(
            f"{prefix}{error.message}",
            validation_error=error,
        )
    return typed_document


@cache
def _validator(schema_name: FlightRequestSchema) -> Validator:
    schema = _load_schema(_SCHEMA_DIRECTORY / f"{schema_name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        registry=_schema_registry(),
        format_checker=FormatChecker(),
    )


@cache
def _schema_registry() -> SchemaRegistry:
    registry: SchemaRegistry = Registry()
    for directory in _SCHEMA_DIRECTORIES:
        for path in sorted(directory.glob("*.schema.json")):
            contents = _load_schema(path)
            if isinstance(contents, bool):
                raise FlightContractError("Flight schema must be a JSON object")
            mapping = cast(Mapping[str, object], contents)
            schema_id = mapping.get("$id")
            if not isinstance(schema_id, str):
                raise FlightContractError("Flight schema must define a string $id")
            registry = registry.with_resource(
                schema_id,
                Resource.from_contents(contents),
            )
    return registry


def _load_schema(path: Path) -> Schema:
    with open(path, encoding="utf-8") as source:
        document: object = json.load(source)
    if not isinstance(document, Mapping):
        raise FlightContractError("Flight schema must be a JSON object")
    return cast(Schema, document)


__all__ = [
    "FlightContractError",
    "FlightRequestSchema",
    "validate_request_document",
]
