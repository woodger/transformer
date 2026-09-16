from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import Schema, SchemaRegistry

from app.contracts.json_types import JsonValue


class SemanticSchemaError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        path: str = "",
        validation_error: ValidationError | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.validation_error = validation_error


_SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")


def validate_schema(document: object, schema_name: str) -> None:
    validator = _validator(schema_name)
    errors = sorted(
        validator.iter_errors(cast(JsonValue, document)),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
        ),
    )
    if not errors:
        return

    error = errors[0]
    path = _json_pointer(error.absolute_path)
    prefix = f"{path}: " if path else ""
    raise SemanticSchemaError(
        f"{prefix}{error.message}",
        path=path,
        validation_error=error,
    )


@cache
def _validator(schema_name: str) -> Validator:
    path = _SCHEMA_DIRECTORY / f"{schema_name}.schema.json"
    if not path.is_file():
        raise ValueError(f"unknown semantic schema: {schema_name}")
    schema = _load_schema(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        registry=_schema_registry(),
        format_checker=FormatChecker(),
    )


@cache
def _schema_registry() -> SchemaRegistry:
    registry: SchemaRegistry = Registry()
    for path in sorted(_SCHEMA_DIRECTORY.glob("*.schema.json")):
        contents = _load_schema(path)
        if isinstance(contents, bool):
            raise SemanticSchemaError("semantic schema must be a JSON object")
        mapping = cast(Mapping[str, object], contents)
        schema_id = mapping.get("$id")
        if not isinstance(schema_id, str):
            raise SemanticSchemaError("semantic schema has no string $id")
        registry = registry.with_resource(
            schema_id,
            Resource.from_contents(contents),
        )
    return registry


def _load_schema(path: Path) -> Schema:
    with open(path, encoding="utf-8") as source:
        document: object = json.load(source)
    if not isinstance(document, Mapping):
        raise SemanticSchemaError("semantic schema must be a JSON object")
    return cast(Schema, document)


def _json_pointer(parts: Iterable[object]) -> str:
    return "".join(
        "/" + str(part).replace("~", "~0").replace("/", "~1")
        for part in parts
    )


__all__ = ["SemanticSchemaError", "validate_schema"]
