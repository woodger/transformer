from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import cast

import rfc8785

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.semantic.v6.constants import OBJECTIVE_LANGUAGE_REVISION


def jcs_sha256(value: JsonValue) -> str:
    return hashlib.sha256(rfc8785.dumps(_binary64_json(value))).hexdigest()


def _binary64_json(value: JsonValue) -> JsonValue:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        if -(2**53 - 1) <= value <= 2**53 - 1:
            return value
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ValueError(
                "JSON numbers must be finite IEEE 754 binary64"
            ) from exc
        if not math.isfinite(converted):
            raise ValueError("JSON numbers must be finite IEEE 754 binary64")
        return converted
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite IEEE 754 binary64")
        return value
    if isinstance(value, list):
        return [_binary64_json(item) for item in value]
    return {key: _binary64_json(item) for key, item in value.items()}


def target_objective_digests(
    model_contract: Mapping[str, object],
) -> JsonObject:
    """Вычислить межпроектные D1-идентификаторы target и objective."""

    target_preimage: JsonObject = {
        "objectiveLanguageRevision": OBJECTIVE_LANGUAGE_REVISION,
        "targetContract": cast(JsonValue, model_contract["targetContract"]),
    }
    objective_preimage: JsonObject = {
        "objectiveLanguageRevision": OBJECTIVE_LANGUAGE_REVISION,
        "objective": cast(JsonValue, model_contract["objective"]),
    }
    target_digest = jcs_sha256(target_preimage)
    objective_digest = jcs_sha256(objective_preimage)
    return {
        "targetContractSha256": target_digest,
        "objectiveSha256": objective_digest,
    }


__all__ = ["jcs_sha256", "target_objective_digests"]
