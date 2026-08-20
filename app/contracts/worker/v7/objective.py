from __future__ import annotations

import hashlib

import rfc8785

from app.contracts.json_types import JsonObject
from app.contracts.ml import (
    CHECKPOINT_FORMAT,
    DATA_CONTRACT_ID,
    DATA_CONTRACT_VERSION,
    OBJECTIVE_ID,
    PREDICTION_SCHEMA_ID,
    TARGET_IDENTITIES,
    TARGET_SCHEMA_ID,
    TARGET_WIDTH,
    TRAINING_RECOVERY_FORMAT,
)
from app.contracts.worker.v7.config import TrainConfig

DIRECT_LOSSES = (
    ("L0", TARGET_IDENTITIES[0], "smooth_l1"),
    ("L1", TARGET_IDENTITIES[1], "smooth_l1"),
    ("L2", TARGET_IDENTITIES[2], "binary_cross_entropy_with_logits"),
    ("L3", TARGET_IDENTITIES[3], "binary_cross_entropy_with_logits"),
    ("L4", TARGET_IDENTITIES[4], "log_mse"),
    ("L5", TARGET_IDENTITIES[5], "binary_cross_entropy_with_logits"),
)


def objective_config(config: TrainConfig) -> JsonObject:
    selection = config.selection
    return {
        "schemaVersion": 2,
        "objectiveId": OBJECTIVE_ID,
        "directLosses": [
            {
                "id": loss_id,
                "targetIndex": index,
                "semantic": semantic,
                "type": loss_type,
                "normalization": "global_row_mean",
                "weight": config.direct_loss_weights[index],
            }
            for index, (loss_id, semantic, loss_type) in enumerate(DIRECT_LOSSES)
        ],
        "auxiliaryLosses": [
            {
                "id": "gaussianNll",
                "type": "gaussian_nll",
                "privateHead": "returnScale",
                "weight": 1.0,
            },
            {
                "id": "expectedValue",
                "type": "take_profit_minus_stop_loss",
                "weight": 0.3,
                "riskPenalty": 0.1,
            },
        ],
        "stagePolicy": {
            "schedule": config.loss_schedule,
            "stageSize": config.stage_size,
            "maximumStage": config.loss_stage,
            "stages": {
                "1": ["L0", "L1", "gaussianNll"],
                "2": [
                    "L0", "L1", "L2", "L3", "L5", "gaussianNll"
                ],
                "3": [
                    "L0", "L1", "L2", "L3", "L5", "gaussianNll",
                    "expectedValue",
                ],
                "4": [
                    "L0", "L1", "L2", "L3", "L4", "L5", "gaussianNll",
                    "expectedValue",
                ],
            },
        },
        "selection": {
            "enabled": selection is not None,
            "aggregation": "global_row_mean",
            "weights": list(config.direct_loss_weights),
            "minDelta": 0.0 if selection is None else selection.min_delta,
            "patience": 0 if selection is None else selection.patience,
            "stagePolicy": "maximum_only_reset",
            "tiePolicy": "earliest",
            "baselinePolicy": "none",
            "invalidScorePolicy": "fail_training",
            "fallbackPolicy": (
                "last_maximum_stage_checkpoint"
                if selection is None
                else "best_selection_score"
            ),
        },
    }


def objective_config_sha256(config: TrainConfig) -> str:
    payload = rfc8785.dumps(objective_config(config))
    return hashlib.sha256(payload).hexdigest()


def ml_contract(config: TrainConfig) -> JsonObject:
    return {
        "targetSchemaId": TARGET_SCHEMA_ID,
        "predictionSchemaId": PREDICTION_SCHEMA_ID,
        "objectiveId": OBJECTIVE_ID,
        "objectiveConfigSha256": objective_config_sha256(config),
        "checkpointFormat": CHECKPOINT_FORMAT,
        "targetWidth": TARGET_WIDTH,
        "predictionSpace": "target",
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "DATA_CONTRACT_ID",
    "DATA_CONTRACT_VERSION",
    "DIRECT_LOSSES",
    "OBJECTIVE_ID",
    "PREDICTION_SCHEMA_ID",
    "TARGET_IDENTITIES",
    "TARGET_SCHEMA_ID",
    "TARGET_WIDTH",
    "TRAINING_RECOVERY_FORMAT",
    "ml_contract",
    "objective_config",
    "objective_config_sha256",
]
