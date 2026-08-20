from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from app.contracts.worker.v7.config import ModelConfig, TrainConfig
from app.contracts.worker.v7.objective import (
    ml_contract,
    objective_config,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode
from app.service.domain.json_types import JsonObject
from app.service.domain.records import PublishedModelRecord


def verify_model_semantics(
    model: PublishedModelRecord,
    *,
    data_contract: JsonObject | None = None,
    requested_ml_contract: JsonObject | None = None,
) -> ModelConfig:
    if model.data_contract is None or model.ml_contract is None:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model generation belongs to another ML contract",
        )
    if data_contract is not None and model.data_contract != data_contract:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model data contract does not match prediction input",
        )
    if requested_ml_contract is not None and (
        model.ml_contract != requested_ml_contract
    ):
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model ML contract does not match prediction job",
        )

    try:
        model_config = ModelConfig.from_dict(model.metadata["model_config"])
        train_config = TrainConfig.from_dict(model.metadata["train_config"])
        if model_config is None or train_config is None:
            raise ValueError("published model configuration is unavailable")
        expected_ml_contract = ml_contract(train_config)
        expected_objective = objective_config(train_config)
        checkpoint = _object(
            model.metadata["checkpoint"],
            "published model checkpoint metadata",
        )
        consistent = (
            model.metadata["data_contract"] == model.data_contract
            and model.metadata["ml_contract"] == expected_ml_contract
            and model.metadata["objective_config"] == expected_objective
            and model.ml_contract == expected_ml_contract
            and model.objective_config_sha256
            == expected_ml_contract["objectiveConfigSha256"]
            and checkpoint["mlContract"] == expected_ml_contract
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


def _object(value: object, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} keys must be strings")
    return cast(JsonObject, dict(mapping))


__all__ = ["verify_model_semantics"]
