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
    parent_data_contract_sha256: str,
    data_contract_sha256: str,
) -> JsonObject:
    initialization: JsonObject = {
        "kind": "publishedModel",
        "parentModelRef": model_ref,
        "parentCheckpointSha256": checkpoint_sha256,
        "parentDataContractSha256": parent_data_contract_sha256,
        "dataContractSha256": data_contract_sha256,
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
    published_model_fields = {
        "kind",
        "parentModelRef",
        "parentCheckpointSha256",
        "parentDataContractSha256",
        "dataContractSha256",
    }
    if kind != "publishedModel" or set(document) != published_model_fields:
        raise ValueError("model initialization is invalid")
    model_ref = document.get("parentModelRef")
    checkpoint_sha256 = document.get("parentCheckpointSha256")
    if not isinstance(model_ref, str) or _MODEL_REF.fullmatch(model_ref) is None:
        raise ValueError("parent model reference is invalid")
    checkpoint_sha256 = _digest(
        checkpoint_sha256,
        "parent checkpoint digest",
    )
    result: JsonObject = {
        "kind": "publishedModel",
        "parentModelRef": model_ref,
        "parentCheckpointSha256": checkpoint_sha256,
        "parentDataContractSha256": _digest(
            document.get("parentDataContractSha256"),
            "parent data contract digest",
        ),
        "dataContractSha256": _digest(
            document.get("dataContractSha256"),
            "data contract digest",
        ),
    }
    return result


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


__all__ = [
    "published_model_initialization",
    "random_initialization",
    "validate_initialization",
]
