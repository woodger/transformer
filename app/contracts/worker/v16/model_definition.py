from __future__ import annotations

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v4 import ModelContract
from app.contracts.semantic.v4.constants import OBJECTIVE_LANGUAGE_REVISION
from app.contracts.semantic.v4.digests import jcs_sha256
from app.contracts.worker.v16.model_config import ModelConfig

_MODEL_IMPLEMENTATION_REVISION = 1


def resolved_semantic_digests(
    model_contract: ModelContract,
    data_contract_sha256: str,
    model_config: ModelConfig,
) -> JsonObject:
    """Выпустить принадлежащую provider-у identity определения модели.

    Identity target-а и objective — межпроектные D1-значения. Определение
    модели дополнительно связывает revision реализации и resolved
    configuration Transformer, поэтому его preimage вычисляет только provider.
    """

    if not _sha256(data_contract_sha256):
        raise ValueError("data contract digest is invalid")

    target_objective = model_contract.target_objective_digests()
    model_definition = jcs_sha256({
        "modelImplementationRevision": _MODEL_IMPLEMENTATION_REVISION,
        "objectiveLanguageRevision": OBJECTIVE_LANGUAGE_REVISION,
        "modelConfig": model_config.to_manifest(),
        "targetContractSha256": target_objective["targetContractSha256"],
        "objectiveSha256": target_objective["objectiveSha256"],
    })
    return {
        "dataContractSha256": data_contract_sha256,
        **target_objective,
        "modelDefinitionSha256": model_definition,
    }


def _sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = ["resolved_semantic_digests"]
