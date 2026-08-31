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

_SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
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
})


class FlightContractError(ValueError):
    """A Flight document does not satisfy its versioned JSON Schema."""

    def __init__(
        self,
        message: str,
        *,
        validation_error: ValidationError | None = None,
    ) -> None:
        super().__init__(message)
        self.validation_error = validation_error


def validate_request_document(
    document: object,
    schema_name: FlightRequestSchema,
) -> JsonObject:
    if schema_name not in _SCHEMA_NAMES:
        raise ValueError(f"unknown Flight contract schema: {schema_name}")
    if not isinstance(document, dict):
        raise FlightContractError("Flight request must be a JSON object")
    typed_document = cast(JsonObject, document)
    validator = _validator(schema_name)
    errors = sorted(
        validator.iter_errors(typed_document),
        key=_error_sort_key,
    )
    if errors:
        error = errors[0]
        raise FlightContractError(
            _format_error(error),
            validation_error=error,
        )
    return typed_document


def _error_sort_key(
    error: ValidationError,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    return (
        int(error.validator in {"allOf", "anyOf", "not", "oneOf"}),
        tuple(str(part) for part in error.absolute_path),
        tuple(str(part) for part in error.absolute_schema_path),
    )


def _format_error(error: ValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path)
    if error.validator == "const" and location:
        return f"{location} must be {_display_value(error.validator_value)}"
    prefix = f"{location}: " if location else ""
    return f"{prefix}{error.message}"


def _display_value(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


@cache
def _validator(schema_name: FlightRequestSchema) -> Validator:
    schema = _load_schema(schema_name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        registry=_schema_registry(),
        format_checker=FormatChecker(),
    )


def _load_schema(name: str) -> Schema:
    with open(_SCHEMA_DIRECTORY / f"{name}.schema.json", encoding="utf-8") as source:
        document: object = json.load(source)
    if not isinstance(document, Mapping):
        raise FlightContractError("Flight schema must be a JSON object")
    return cast(Schema, document)


@cache
def _schema_registry() -> SchemaRegistry:
    registry: SchemaRegistry = Registry()
    for path in sorted(_SCHEMA_DIRECTORY.glob("*.schema.json")):
        with open(path, encoding="utf-8") as source:
            document: object = json.load(source)
        if not isinstance(document, Mapping):
            raise FlightContractError("Flight schema must be a JSON object")
        contents = cast(Schema, document)
        if isinstance(contents, bool):
            raise FlightContractError("Flight schema must be a JSON object")
        schema_id = contents.get("$id")
        if not isinstance(schema_id, str):
            raise FlightContractError("Flight schema must define a string $id")
        registry = registry.with_resource(
            schema_id,
            Resource.from_contents(contents),
        )
    return registry


__all__ = [
    "FlightContractError",
    "FlightRequestSchema",
    "validate_request_document",
]
