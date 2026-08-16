from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.worker.v5 import WorkerContractError


def object_document(value: object, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise WorkerContractError(f"worker {label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise WorkerContractError(f"worker {label} field names must be strings")
    return cast(JsonObject, dict(mapping))


def object_field(document: JsonObject, name: str) -> JsonObject:
    return object_document(document.get(name), name)


def object_list(value: object, label: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise WorkerContractError(f"worker {label} must be an array")
    return [
        object_document(item, label)
        for item in cast(list[object], value)
    ]


def string_field(document: JsonObject, name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str):
        raise WorkerContractError(f"worker {name} must be a string")
    return value


def optional_string_field(document: JsonObject, name: str) -> str | None:
    value = document.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkerContractError(f"worker {name} must be a string or null")
    return value


def integer_field(document: JsonObject, name: str) -> int:
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerContractError(f"worker {name} must be an integer")
    return value


def boolean_field(document: JsonObject, name: str) -> bool:
    value = document.get(name)
    if not isinstance(value, bool):
        raise WorkerContractError(f"worker {name} must be a boolean")
    return value


__all__ = [
    "boolean_field",
    "integer_field",
    "object_document",
    "object_field",
    "object_list",
    "optional_string_field",
    "string_field",
]
