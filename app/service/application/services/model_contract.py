from __future__ import annotations

from collections.abc import Mapping

from app.contracts.semantic.v6 import ModelContract, SemanticContractError
from app.contracts.worker.v21.config import ModelConfig
from app.contracts.worker.v21.constants import CHECKPOINT_FORMAT
from app.contracts.worker.v21.model_definition import resolved_semantic_digests
from app.service.domain.errors import ServiceError
from app.service.domain.initialization import validate_initialization
from app.service.domain.job import ErrorCode
from app.service.domain.json_types import JsonObject
from app.service.domain.records import PublishedModelRecord

_DIGEST_LAYERS = (
    ("data", "dataContractSha256"),
    ("target", "targetContractSha256"),
    ("objective", "objectiveSha256"),
    ("model", "modelDefinitionSha256"),
)


def verify_model_integrity(
    model: PublishedModelRecord,
) -> ModelConfig:
    try:
        contract = ModelContract.from_document(model.model_contract)
        data_digest = _digest(
            model.data_contract.get("dataContractSha256"),
            "published model data contract digest",
        )
        metadata = model.metadata
        model_config = ModelConfig.from_manifest(metadata.get("modelConfig"))
        calculated = resolved_semantic_digests(
            contract,
            data_digest,
            model_config,
        )
        if calculated != model.semantic_digests:
            _raise_corrupt_digest(model.semantic_digests, calculated)

        if (
            metadata.get("format") != CHECKPOINT_FORMAT
            or metadata.get("dataContract") != model.data_contract
            or metadata.get("modelContract") != model.model_contract
            or metadata.get("predictionDefinition")
            != contract.prediction_definition(
                _positive_integer(model.data_contract.get("seqLen"))
            )
            or metadata.get("semanticDigests") != model.semantic_digests
        ):
            raise ValueError("checkpoint metadata differs from published model")
        model_initialization(model)
    except ServiceError:
        raise
    except (KeyError, SemanticContractError, TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCode.MODEL_CORRUPT,
            "published model semantic metadata is invalid",
            detail={
                "code": "MODEL_CORRUPT",
                "reason": "STORED_CONTRACT_INVALID",
                "layer": "model",
                "path": "",
                "message": "published model semantic metadata is invalid",
            },
        ) from exc
    return model_config


def verify_model_for_predict(
    model: PublishedModelRecord,
    *,
    data_contract: JsonObject,
) -> ModelConfig:
    model_config = verify_model_integrity(model)
    _verify_data_compatibility(model.semantic_digests, data_contract)
    return model_config


def verify_parent_model_for_fit(
    model: PublishedModelRecord,
    *,
    model_config: ModelConfig | None,
    model_contract: JsonObject,
    semantic_digests: JsonObject,
) -> ModelConfig:
    parent_model_config = verify_model_integrity(model)
    _verify_compatible(model.semantic_digests, semantic_digests)
    if model.model_contract != model_contract:
        raise ServiceError(
            ErrorCode.MODEL_CORRUPT,
            "equal semantic digests identify different canonical documents",
        )
    if model_config != parent_model_config:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "parent model configuration does not match fit job",
        )
    return parent_model_config


def model_initialization(model: PublishedModelRecord) -> JsonObject:
    try:
        return validate_initialization(model.metadata.get("initialization"))
    except ValueError as exc:
        raise ServiceError(
            ErrorCode.MODEL_CORRUPT,
            "published model initialization metadata is invalid",
        ) from exc


def _verify_compatible(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> None:
    for layer, field in _DIGEST_LAYERS:
        expected_digest = _digest(expected.get(field), f"stored {layer} digest")
        actual_digest = _digest(actual.get(field), f"requested {layer} digest")
        if expected_digest != actual_digest:
            message = f"model {layer} contract does not match the requested job"
            raise ServiceError(
                ErrorCode.MODEL_SCHEMA_MISMATCH,
                message,
                detail={
                    "code": "MODEL_SCHEMA_MISMATCH",
                    "reason": "DIGEST_MISMATCH",
                    "layer": layer,
                    "expectedSha256": expected_digest,
                    "actualSha256": actual_digest,
                    "message": message,
                },
            )


def _verify_data_compatibility(
    expected: Mapping[str, object],
    data_contract: Mapping[str, object],
) -> None:
    expected_digest = _digest(
        expected.get("dataContractSha256"),
        "stored data digest",
    )
    actual_digest = _digest(
        data_contract.get("dataContractSha256"),
        "requested data digest",
    )
    if expected_digest == actual_digest:
        return
    message = "model data contract does not match the requested job"
    raise ServiceError(
        ErrorCode.MODEL_SCHEMA_MISMATCH,
        message,
        detail={
            "code": "MODEL_SCHEMA_MISMATCH",
            "reason": "DIGEST_MISMATCH",
            "layer": "data",
            "expectedSha256": expected_digest,
            "actualSha256": actual_digest,
            "message": message,
        },
    )


def _raise_corrupt_digest(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> None:
    for layer, field in _DIGEST_LAYERS:
        expected_digest = _digest(expected.get(field), f"stored {layer} digest")
        actual_digest = _digest(actual.get(field), f"calculated {layer} digest")
        if expected_digest != actual_digest:
            message = f"published model {layer} digest is inconsistent"
            raise ServiceError(
                ErrorCode.MODEL_CORRUPT,
                message,
                detail={
                    "code": "MODEL_CORRUPT",
                    "reason": "DIGEST_MISMATCH",
                    "layer": layer,
                    "expectedSha256": expected_digest,
                    "actualSha256": actual_digest,
                    "message": message,
                },
            )
    raise ValueError("semantic digest document is invalid")


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _positive_integer(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("positive integer is invalid")
    return value


__all__ = [
    "model_initialization",
    "verify_model_for_predict",
    "verify_model_integrity",
    "verify_parent_model_for_fit",
]
