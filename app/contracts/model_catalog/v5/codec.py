from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError
from jsonschema.protocols import Validator
from referencing import Registry, Resource
from referencing.jsonschema import Schema, SchemaRegistry

from app.contracts.json_types import JsonObject


class ModelCatalogContractError(ValueError):
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
    _CONTRACTS_DIRECTORY / "semantic" / "v4" / "schemas",
    _CONTRACTS_DIRECTORY / "checkpoint" / "v10" / "schemas",
)
_SCHEMA_NAMES = frozenset({
    "detail-request",
    "detail-result",
    "error-detail",
    "fixture-manifest",
    "list-request",
    "list-result",
})


def validate_catalog_document(
    document: object,
    schema_name: str,
) -> JsonObject:
    if schema_name not in _SCHEMA_NAMES:
        raise ValueError(f"unknown model catalog schema: {schema_name}")
    if not isinstance(document, dict):
        raise ModelCatalogContractError(
            "model catalog document must be a JSON object"
        )
    typed = cast(JsonObject, document)
    errors = sorted(
        _validator(schema_name).iter_errors(typed),
        key=lambda error: (
            int(error.validator in {"allOf", "anyOf", "not", "oneOf"}),
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
        ),
    )
    if errors:
        error = errors[0]
        path = "/" + "/".join(
            str(part).replace("~", "~0").replace("/", "~1")
            for part in error.absolute_path
        )
        prefix = f"{path}: " if path != "/" else ""
        raise ModelCatalogContractError(
            f"{prefix}{error.message}",
            validation_error=error,
        )
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
                raise ModelCatalogContractError(
                    "model catalog schema must be an object"
                )
            identifier = cast(Mapping[str, object], contents).get("$id")
            if not isinstance(identifier, str):
                raise ModelCatalogContractError(
                    f"model catalog schema has no $id: {path}"
                )
            registry = registry.with_resource(
                identifier,
                Resource.from_contents(contents),
            )
    return registry


def _load_schema(path: Path) -> Schema:
    with path.open(encoding="utf-8") as source:
        document: object = json.load(source)
    if not isinstance(document, Mapping):
        raise ModelCatalogContractError(
            "model catalog schema must be a JSON object"
        )
    return cast(Schema, document)


__all__ = ["ModelCatalogContractError", "validate_catalog_document"]
