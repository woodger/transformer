from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from app.contracts.worker.v10.config import ModelConfig, TrainConfig
from app.contracts.worker.v10.objective import (
    objective_config_sha256,
    objective_from_ml_contract,
)
from app.service.domain.errors import ServiceError
from app.service.domain.initialization import validate_initialization
from app.service.domain.job import ErrorCode
from app.service.domain.json_types import JsonObject
from app.service.domain.records import PublishedModelRecord


def verify_model_integrity(model: PublishedModelRecord) -> ModelConfig:
    data_contract, ml_contract = _model_contracts(model)

    return _verify_model_integrity(
        model,
        data_contract=data_contract,
        ml_contract=ml_contract,
    )


def verify_model_for_predict(
    model: PublishedModelRecord,
    *,
    data_contract: JsonObject,
    ml_contract: JsonObject,
) -> ModelConfig:
    model_data_contract, model_ml_contract = _model_contracts(model)
    if model_data_contract != data_contract:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model data contract does not match the requested job",
        )
    if model_ml_contract != ml_contract:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model ML contract does not match the requested job",
        )

    return _verify_model_integrity(
        model,
        data_contract=model_data_contract,
        ml_contract=model_ml_contract,
    )


def verify_parent_model_for_fit(
    model: PublishedModelRecord,
    *,
    model_config: ModelConfig | None,
    data_contract: JsonObject,
    ml_contract: JsonObject,
) -> ModelConfig:
    parent_data_contract, parent_ml_contract = _model_contracts(model)
    if parent_ml_contract != ml_contract:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model ML contract does not match the requested job",
        )

    parent_model_config = _verify_model_integrity(
        model,
        data_contract=parent_data_contract,
        ml_contract=parent_ml_contract,
    )
    parent_digest = parent_data_contract.get("data_contract_sha256")
    current_digest = data_contract.get("data_contract_sha256")
    if parent_digest != current_digest:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "parent model data contract is not compatible with fit job",
        )
    if model_config != parent_model_config:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "parent model configuration does not match fit job",
        )
    return parent_model_config


def _model_contracts(
    model: PublishedModelRecord,
) -> tuple[JsonObject, JsonObject]:
    if model.data_contract is None or model.ml_contract is None:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model generation belongs to another ML contract",
        )
    return model.data_contract, model.ml_contract


def _verify_model_integrity(
    model: PublishedModelRecord,
    *,
    data_contract: JsonObject,
    ml_contract: JsonObject,
) -> ModelConfig:
    try:
        model_config = ModelConfig.from_dict(model.metadata["model_config"])
        train_config = TrainConfig.from_dict(model.metadata["train_config"])
        if model_config is None or train_config is None:
            raise ValueError("published model configuration is unavailable")
        objective = objective_from_ml_contract(ml_contract)
        checkpoint = _object(
            model.metadata["checkpoint"],
            "published model checkpoint metadata",
        )
        initialization = model_initialization(model)
        checkpoint_initialization = validate_initialization(
            checkpoint.get("initialization"),
            missing_is_random=True,
        )
        consistent = (
            model.metadata["data_contract"] == data_contract
            and model.metadata["ml_contract"] == ml_contract
            and model.metadata["objective"] == objective.to_document()
            and model_config.out_dim == objective.target_width
            and model.objective_config_sha256
            == objective_config_sha256(objective)
            and checkpoint["mlContract"] == ml_contract
            and checkpoint_initialization == initialization
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ServiceError(
            ErrorCode.MODEL_CORRUPT,
            "published model semantic metadata is invalid",
        ) from exc
    if not consistent:
        raise ServiceError(
            ErrorCode.MODEL_CORRUPT,
            "published model semantic metadata is inconsistent",
        )
    return model_config


def model_initialization(model: PublishedModelRecord) -> JsonObject:
    try:
        return validate_initialization(
            model.metadata.get("initialization"),
            missing_is_random=True,
        )
    except ValueError as exc:
        raise ServiceError(
            ErrorCode.MODEL_CORRUPT,
            "published model initialization metadata is invalid",
        ) from exc


def _object(value: object, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} keys must be strings")
    return cast(JsonObject, dict(mapping))


__all__ = [
    "model_initialization",
    "verify_model_for_predict",
    "verify_model_integrity",
    "verify_parent_model_for_fit",
]
