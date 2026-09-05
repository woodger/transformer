from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import cast

import rfc8785

from app.contracts.json_types import JsonObject, JsonValue


def jcs_sha256(value: JsonValue) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def semantic_digests(
    model_contract: Mapping[str, object],
    data_contract_sha256: str,
) -> JsonObject:
    revision = cast(JsonValue, model_contract["objectiveLanguageRevision"])
    target_preimage: JsonObject = {
        "objectiveLanguageRevision": revision,
        "targetContract": cast(JsonValue, model_contract["targetContract"]),
    }
    objective_preimage: JsonObject = {
        "objectiveLanguageRevision": revision,
        "objective": cast(JsonValue, model_contract["objective"]),
    }
    target_digest = jcs_sha256(target_preimage)
    objective_digest = jcs_sha256(objective_preimage)
    model_preimage: JsonObject = {
        "objectiveLanguageRevision": revision,
        "modelConfig": cast(JsonValue, model_contract["modelConfig"]),
        "targetContractSha256": target_digest,
        "objectiveSha256": objective_digest,
    }
    return {
        "dataContractSha256": data_contract_sha256,
        "targetContractSha256": target_digest,
        "objectiveSha256": objective_digest,
        "modelContractSha256": jcs_sha256(model_preimage),
    }


__all__ = ["jcs_sha256", "semantic_digests"]
