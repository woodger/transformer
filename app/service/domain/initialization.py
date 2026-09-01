from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

from app.service.domain.json_types import JsonObject

_MODEL_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def random_initialization() -> JsonObject:
    return {"kind": "random"}


def published_model_initialization(
    model_ref: str,
    checkpoint_sha256: str,
) -> JsonObject:
    initialization: JsonObject = {
        "kind": "publishedModel",
        "parentModelRef": model_ref,
        "parentCheckpointSha256": checkpoint_sha256,
    }
    return validate_initialization(initialization)


def validate_initialization(
    value: object,
    *,
    missing_is_random: bool = False,
) -> JsonObject:
    if value is None and missing_is_random:
        return random_initialization()
    if not isinstance(value, Mapping):
        raise ValueError("model initialization must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError("model initialization keys must be strings")
    document = cast(dict[str, object], dict(mapping))
    kind = document.get("kind")
    if kind == "random" and set(document) == {"kind"}:
        return random_initialization()
    if kind != "publishedModel" or set(document) != {
        "kind",
        "parentModelRef",
        "parentCheckpointSha256",
    }:
        raise ValueError("model initialization is invalid")
    model_ref = document.get("parentModelRef")
    checkpoint_sha256 = document.get("parentCheckpointSha256")
    if not isinstance(model_ref, str) or _MODEL_REF.fullmatch(model_ref) is None:
        raise ValueError("parent model reference is invalid")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise ValueError("parent checkpoint digest is invalid")
    return {
        "kind": "publishedModel",
        "parentModelRef": model_ref,
        "parentCheckpointSha256": checkpoint_sha256,
    }


__all__ = [
    "published_model_initialization",
    "random_initialization",
    "validate_initialization",
]
