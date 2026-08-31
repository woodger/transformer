from __future__ import annotations

from collections.abc import Sequence

from app.contracts.json_types import JsonObject

DATA_CONTRACT_ID = "inventory.learning-dataset"
DATA_CONTRACT_VERSION = 2
TARGET_SCHEMA_ID = "inventory.target.v2"
PREDICTION_SCHEMA_ID = "transformer.prediction.target-aligned.v3"
OBJECTIVE_ID = "transformer.objective.declarative.v1"
CHECKPOINT_FORMAT = "transformer-checkpoint-v5"
TRAINING_RECOVERY_FORMAT = "transformer-training-recovery-v5"

TARGET_IDENTITIES = (
    "MeanReturn",
    "SigmaReturn",
    "ProbTP",
    "ProbSL",
    "VolatilityNext",
    "HittingProbTP",
)
MAX_TARGET_WIDTH = len(TARGET_IDENTITIES)


def canonical_targets(values: Sequence[str]) -> tuple[str, ...]:
    targets = tuple(values)
    if not targets:
        raise ValueError("targets must contain at least one target")
    if len(targets) > MAX_TARGET_WIDTH:
        raise ValueError(f"targets must contain at most {MAX_TARGET_WIDTH} targets")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must not contain duplicates")
    if any(target not in TARGET_IDENTITIES for target in targets):
        raise ValueError("targets contain an unsupported target identity")
    expected = tuple(target for target in TARGET_IDENTITIES if target in targets)
    if targets != expected:
        raise ValueError("targets must follow the canonical target order")
    return targets


def target_identity(index: int, name: str) -> JsonObject:
    if isinstance(index, bool) or index < 0:
        raise ValueError("target index must be a non-negative integer")
    if name not in TARGET_IDENTITIES:
        raise ValueError("target name is unsupported")
    return {"index": index, "name": name}


__all__ = [
    "CHECKPOINT_FORMAT",
    "DATA_CONTRACT_ID",
    "DATA_CONTRACT_VERSION",
    "MAX_TARGET_WIDTH",
    "OBJECTIVE_ID",
    "PREDICTION_SCHEMA_ID",
    "TARGET_IDENTITIES",
    "TARGET_SCHEMA_ID",
    "TRAINING_RECOVERY_FORMAT",
    "canonical_targets",
    "target_identity",
]
