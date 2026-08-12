from __future__ import annotations

from app.contracts.worker.v3.config import ModelConfig, TrainConfig
from app.contracts.worker.v3.objective import (
    ml_contract,
    objective_config,
)
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode


def verify_model_semantics(
    model,
    *,
    data_contract: dict | None = None,
    requested_ml_contract: dict | None = None,
) -> ModelConfig:
    if model.data_contract is None or model.ml_contract is None:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "model generation belongs to another ML contract",
        )
    if data_contract is not None and (
        model.data_contract.get("data_contract_sha256")
        != data_contract["data_contract_sha256"]
    ):
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
        expected_ml_contract = ml_contract(train_config)
        expected_objective = objective_config(train_config)
        checkpoint = model.metadata["checkpoint"]
        consistent = (
            model_config is not None
            and train_config is not None
            and model.metadata["data_contract"] == model.data_contract
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


__all__ = ["verify_model_semantics"]
