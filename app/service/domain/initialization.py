from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

from app.service.domain.json_types import JsonObject

_MODEL_REF = re.compile(r"^mdl_[0-9a-f]{32}$")


def random_initialization() -> JsonObject:
    """Return the checkpoint-owned resolved random initialization."""

    return {"source": "random"}


def published_model_initialization(
    model_ref: str,
    checkpoint_sha256: str,
    parent_digests: Mapping[str, object],
    current_digests: Mapping[str, object],
) -> JsonObject:
    initialization: dict[str, object] = {
        "source": "publishedModel",
        "parentModelRef": model_ref,
        "parentCheckpointSha256": checkpoint_sha256,
        "parentDataContractSha256": parent_digests["dataContractSha256"],
        "dataContractSha256": current_digests["dataContractSha256"],
        "parentTargetContractSha256": parent_digests[
            "targetContractSha256"
        ],
        "targetContractSha256": current_digests["targetContractSha256"],
        "parentObjectiveSha256": parent_digests["objectiveSha256"],
        "objectiveSha256": current_digests["objectiveSha256"],
        "parentModelContractSha256": parent_digests[
            "modelContractSha256"
        ],
        "modelContractSha256": current_digests["modelContractSha256"],
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
    source = document.get("source")
    if source == "random" and set(document) == {"source"}:
        return random_initialization()
    published_model_fields = {
        "source",
        "parentModelRef",
        "parentCheckpointSha256",
        "parentDataContractSha256",
        "dataContractSha256",
        "parentTargetContractSha256",
        "targetContractSha256",
        "parentObjectiveSha256",
        "objectiveSha256",
        "parentModelContractSha256",
        "modelContractSha256",
    }
    if source != "publishedModel" or set(document) != published_model_fields:
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
        "source": "publishedModel",
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
        "parentTargetContractSha256": _digest(
            document.get("parentTargetContractSha256"),
            "parent target contract digest",
        ),
        "targetContractSha256": _digest(
            document.get("targetContractSha256"),
            "target contract digest",
        ),
        "parentObjectiveSha256": _digest(
            document.get("parentObjectiveSha256"),
            "parent objective digest",
        ),
        "objectiveSha256": _digest(
            document.get("objectiveSha256"),
            "objective digest",
        ),
        "parentModelContractSha256": _digest(
            document.get("parentModelContractSha256"),
            "parent model contract digest",
        ),
        "modelContractSha256": _digest(
            document.get("modelContractSha256"),
            "model contract digest",
        ),
    }
    return result


def validate_requested_initialization(value: object) -> JsonObject:
    """Validate the Flight intent form before the service resolves lineage."""

    if not isinstance(value, Mapping):
        raise ValueError("requested model initialization must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError("requested model initialization keys must be strings")
    document = cast(dict[str, object], dict(mapping))
    source = document.get("source")
    if source == "random" and set(document) == {"source"}:
        return {"source": "random"}
    if source != "publishedModel" or set(document) != {"source", "modelRef"}:
        raise ValueError("requested model initialization is invalid")
    model_ref = document.get("modelRef")
    if not isinstance(model_ref, str) or _MODEL_REF.fullmatch(model_ref) is None:
        raise ValueError("requested parent model reference is invalid")
    return {"source": "publishedModel", "modelRef": model_ref}


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
    "validate_requested_initialization",
]
