from __future__ import annotations

from app.contracts.json_types import JsonObject

DATA_CONTRACT_ID = "inventory.learning-dataset"
DATA_CONTRACT_VERSION = 2
TARGET_SCHEMA_ID = "inventory.target.v2"
PREDICTION_SCHEMA_ID = "transformer.prediction.target-aligned.v2"
OBJECTIVE_ID = "transformer.objective.target-aligned.v2"
CHECKPOINT_FORMAT = "transformer-checkpoint-v4"
TRAINING_RECOVERY_FORMAT = "transformer-training-recovery-v4"
TARGET_WIDTH = 6

TARGET_IDENTITIES = (
    "MeanReturn",
    "SigmaReturn",
    "ProbTP",
    "ProbSL",
    "VolatilityNext",
    "HittingProbTP",
)


def target_identity(index: int) -> JsonObject:
    if isinstance(index, bool) or not 0 <= index < len(TARGET_IDENTITIES):
        raise ValueError("target index must be between 0 and 5")
    return {"index": index, "name": TARGET_IDENTITIES[index]}


__all__ = [
    "CHECKPOINT_FORMAT",
    "DATA_CONTRACT_ID",
    "DATA_CONTRACT_VERSION",
    "OBJECTIVE_ID",
    "PREDICTION_SCHEMA_ID",
    "TARGET_IDENTITIES",
    "TARGET_SCHEMA_ID",
    "TARGET_WIDTH",
    "TRAINING_RECOVERY_FORMAT",
    "target_identity",
]
