from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import Schema, SchemaRegistry

from app.contracts.json_types import JsonObject

_SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
_CONTRACTS_DIRECTORY = Path(__file__).parents[2]
_SCHEMA_DIRECTORIES = (
    _SCHEMA_DIRECTORY,
    _CONTRACTS_DIRECTORY / "semantic" / "v2" / "schemas",
)
_SCHEMA_NAMES = frozenset({
    "checkpoint-artifact",
    "checkpoint-metadata",
    "recovery-metadata",
    "resolved-initialization",
})


def validate_checkpoint_document(
    document: object,
    schema_name: str,
) -> JsonObject:
    if schema_name not in _SCHEMA_NAMES:
        raise ValueError(f"unknown checkpoint contract schema: {schema_name}")
    if not isinstance(document, dict):
        raise ValueError("checkpoint contract document must be a JSON object")
    typed = cast(JsonObject, document)
    errors = sorted(
        _validator(schema_name).iter_errors(typed),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise ValueError(f"{path}: {error.message}")
    return typed


@cache
def _validator(schema_name: str) -> Validator:
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
                raise ValueError("checkpoint schema must be an object")
            identifier = contents.get("$id")
            if not isinstance(identifier, str):
                raise ValueError(f"checkpoint schema has no $id: {path}")
            registry = registry.with_resource(
                identifier,
                Resource.from_contents(cast(Schema, contents)),
            )
    return registry


def _load_schema(path: Path) -> Schema:
    with path.open(encoding="utf-8") as source:
        return cast(Schema, json.load(source))


__all__ = ["validate_checkpoint_document"]
