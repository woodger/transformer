from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import rfc8785

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.ml import (
    CHECKPOINT_FORMAT,
    DATA_CONTRACT_ID,
    DATA_CONTRACT_VERSION,
    MAX_TARGET_WIDTH,
    OBJECTIVE_ID,
    PREDICTION_SCHEMA_ID,
    TARGET_IDENTITIES,
    TARGET_SCHEMA_ID,
    TRAINING_RECOVERY_FORMAT,
    canonical_targets,
)

OBJECTIVE_SCHEMA_VERSION = 1
OBJECTIVE_AGGREGATION = "WeightedSum"
OBJECTIVE_REDUCTION = "GlobalRowMean"
STATIC_BALANCING_OPERATOR = "Static"

DIRECT_LOSS_OPERATORS = {
    "MeanReturn": "SmoothL1",
    "SigmaReturn": "SmoothL1",
    "ProbTP": "BinaryCrossEntropyWithLogits",
    "ProbSL": "BinaryCrossEntropyWithLogits",
    "VolatilityNext": "LogMSE",
    "HittingProbTP": "BinaryCrossEntropyWithLogits",
}
AUXILIARY_LOSS_OPERATORS = (
    "GaussianNLL",
    "ExpectedValue",
    "RiskAdjustedExpectedValue",
)


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    """Validated declarative learning objective received from a Consumer."""

    targets: tuple[str, ...]
    objective: JsonObject

    @classmethod
    def from_document(cls, document: object) -> ObjectiveConfig:
        wrapper = _object(document, "objective configuration")
        if set(wrapper) != {"targets", "objective"}:
            raise ValueError(
                "objective configuration must contain targets and objective"
            )
        targets = canonical_targets(
            _string_sequence(wrapper["targets"], "targets")
        )
        objective = _object(wrapper["objective"], "objective")
        _validate_objective(targets, objective)
        return cls(targets=targets, objective=cast(JsonObject, objective))

    def to_document(self) -> JsonObject:
        return {
            "targets": list(self.targets),
            "objective": _copy_json_object(self.objective),
        }

    @property
    def target_width(self) -> int:
        return len(self.targets)

    @property
    def direct_losses(self) -> tuple[JsonObject, ...]:
        return tuple(
            _copy_json_object(_object(item, "direct loss"))
            for item in _sequence(self.objective["directLosses"], "directLosses")
        )

    @property
    def auxiliary_losses(self) -> tuple[JsonObject, ...]:
        return tuple(
            _copy_json_object(_object(item, "auxiliary loss"))
            for item in _sequence(
                self.objective["auxiliaryLosses"],
                "auxiliaryLosses",
            )
        )

    @property
    def direct_loss_weights(self) -> tuple[float, ...]:
        return tuple(
            _positive_number(item["weight"], "direct loss weight")
            for item in self.direct_losses
        )

    @property
    def auxiliary_operators(self) -> tuple[str, ...]:
        return tuple(cast(str, item["operator"]) for item in self.auxiliary_losses)

    @property
    def requires_return_scale(self) -> bool:
        operators = frozenset(self.auxiliary_operators)
        return bool(
            operators
            & {"GaussianNLL", "RiskAdjustedExpectedValue"}
        )


def default_objective(
    direct_loss_weights: Sequence[float] = (1.0,) * MAX_TARGET_WIDTH,
) -> ObjectiveConfig:
    weights = tuple(float(weight) for weight in direct_loss_weights)
    if len(weights) != MAX_TARGET_WIDTH:
        raise ValueError(
            f"direct loss weights must contain {MAX_TARGET_WIDTH} values"
        )
    document: JsonObject = {
        "targets": list(TARGET_IDENTITIES),
        "objective": {
            "schemaVersion": OBJECTIVE_SCHEMA_VERSION,
            "aggregation": OBJECTIVE_AGGREGATION,
            "reduction": OBJECTIVE_REDUCTION,
            "directLosses": [
                {
                    "target": target,
                    "operator": DIRECT_LOSS_OPERATORS[target],
                    "weight": weight,
                }
                for target, weight in zip(TARGET_IDENTITIES, weights, strict=True)
            ],
            "auxiliaryLosses": [
                {"operator": "GaussianNLL", "weight": 1.0},
                {
                    "operator": "RiskAdjustedExpectedValue",
                    "weight": 0.3,
                    "riskPenalty": 0.1,
                },
            ],
            "balancing": {"operator": STATIC_BALANCING_OPERATOR},
        },
    }
    return ObjectiveConfig.from_document(document)


def objective_config(config: ObjectiveConfig) -> JsonObject:
    return config.to_document()


def objective_config_sha256(config: ObjectiveConfig) -> str:
    payload = rfc8785.dumps(config.to_document())
    return hashlib.sha256(payload).hexdigest()


def ml_contract(
    config: ObjectiveConfig,
    *,
    target_schema_id: str = TARGET_SCHEMA_ID,
) -> JsonObject:
    if not target_schema_id:
        raise ValueError("target_schema_id must be a non-empty string")
    return {
        "targetSchemaId": target_schema_id,
        "predictionSchemaId": PREDICTION_SCHEMA_ID,
        "objectiveId": OBJECTIVE_ID,
        "objectiveConfigSha256": objective_config_sha256(config),
        "checkpointFormat": CHECKPOINT_FORMAT,
        "targets": list(config.targets),
        "targetWidth": config.target_width,
        "predictionSpace": "target",
        "objective": _copy_json_object(config.objective),
    }


def objective_from_ml_contract(document: object) -> ObjectiveConfig:
    contract = _object(document, "ML contract")
    config = ObjectiveConfig.from_document({
        "targets": contract.get("targets"),
        "objective": contract.get("objective"),
    })
    expected = ml_contract(
        config,
        target_schema_id=_string(contract.get("targetSchemaId"), "targetSchemaId"),
    )
    if contract != expected:
        raise ValueError("ML contract is inconsistent with its objective")
    return config


def _validate_objective(
    targets: tuple[str, ...],
    objective: Mapping[str, object],
) -> None:
    required = {
        "schemaVersion",
        "aggregation",
        "reduction",
        "directLosses",
        "auxiliaryLosses",
        "balancing",
    }
    if set(objective) != required:
        raise ValueError("objective has unsupported or missing fields")
    if objective["schemaVersion"] != OBJECTIVE_SCHEMA_VERSION:
        raise ValueError(
            f"objective schemaVersion must be {OBJECTIVE_SCHEMA_VERSION}"
        )
    if objective["aggregation"] != OBJECTIVE_AGGREGATION:
        raise ValueError(f"objective aggregation must be {OBJECTIVE_AGGREGATION}")
    if objective["reduction"] != OBJECTIVE_REDUCTION:
        raise ValueError(f"objective reduction must be {OBJECTIVE_REDUCTION}")

    balancing = _object(objective["balancing"], "balancing")
    if balancing != {"operator": STATIC_BALANCING_OPERATOR}:
        raise ValueError("objective balancing operator must be Static")

    raw_direct = _sequence(objective["directLosses"], "directLosses")
    if len(raw_direct) != len(targets):
        raise ValueError("directLosses must contain one item for every target")
    for index, (raw_loss, target) in enumerate(
        zip(raw_direct, targets, strict=True)
    ):
        loss = _object(raw_loss, f"directLosses[{index}]")
        if set(loss) != {"target", "operator", "weight"}:
            raise ValueError(
                f"directLosses[{index}] has unsupported or missing fields"
            )
        if loss["target"] != target:
            raise ValueError("directLosses must follow targets in canonical order")
        if loss["operator"] != DIRECT_LOSS_OPERATORS[target]:
            raise ValueError(f"unsupported direct loss operator for {target}")
        _positive_number(loss["weight"], f"directLosses[{index}].weight")

    raw_auxiliary = _sequence(objective["auxiliaryLosses"], "auxiliaryLosses")
    seen: set[str] = set()
    for index, raw_loss in enumerate(raw_auxiliary):
        loss = _object(raw_loss, f"auxiliaryLosses[{index}]")
        operator = _string(
            loss.get("operator"),
            f"auxiliaryLosses[{index}].operator",
        )
        if operator not in AUXILIARY_LOSS_OPERATORS:
            raise ValueError(f"unsupported auxiliary loss operator: {operator}")
        if operator in seen:
            raise ValueError("auxiliaryLosses must not contain duplicates")
        seen.add(operator)
        required_fields = {"operator", "weight"}
        if operator == "RiskAdjustedExpectedValue":
            required_fields.add("riskPenalty")
        if set(loss) != required_fields:
            raise ValueError(
                f"auxiliaryLosses[{index}] has unsupported or missing fields"
            )
        _positive_number(loss["weight"], f"auxiliaryLosses[{index}].weight")
        if operator == "RiskAdjustedExpectedValue":
            _positive_number(
                loss["riskPenalty"],
                f"auxiliaryLosses[{index}].riskPenalty",
            )

    selected = frozenset(targets)
    if "GaussianNLL" in seen and "MeanReturn" not in selected:
        raise ValueError("GaussianNLL requires MeanReturn")
    for operator in ("ExpectedValue", "RiskAdjustedExpectedValue"):
        if operator in seen and not {"ProbTP", "ProbSL"} <= selected:
            raise ValueError(f"{operator} requires ProbTP and ProbSL")
    if "RiskAdjustedExpectedValue" in seen and "GaussianNLL" not in seen:
        raise ValueError(
            "RiskAdjustedExpectedValue requires GaussianNLL and MeanReturn"
        )
    if {"ExpectedValue", "RiskAdjustedExpectedValue"} <= seen:
        raise ValueError(
            "ExpectedValue and RiskAdjustedExpectedValue are mutually exclusive"
        )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} field names must be strings")
    return {cast(str, key): item for key, item in mapping.items()}


def _copy_json_object(value: Mapping[str, object]) -> JsonObject:
    return {
        key: _copy_json_value(item)
        for key, item in value.items()
    }


def _copy_json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise ValueError("JSON object field names must be strings")
        return {
            cast(str, key): _copy_json_value(item)
            for key, item in mapping.items()
        }
    if isinstance(value, list):
        return [_copy_json_value(item) for item in cast(list[object], value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("objective contains a non-JSON value")


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return tuple(cast(Sequence[object], value))


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    values = _sequence(value, label)
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"{label} must contain only strings")
    return cast(tuple[str, ...], values)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be a positive number")
    return parsed


__all__ = [
    "AUXILIARY_LOSS_OPERATORS",
    "CHECKPOINT_FORMAT",
    "DATA_CONTRACT_ID",
    "DATA_CONTRACT_VERSION",
    "DIRECT_LOSS_OPERATORS",
    "MAX_TARGET_WIDTH",
    "OBJECTIVE_ID",
    "OBJECTIVE_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_ID",
    "TARGET_IDENTITIES",
    "TARGET_SCHEMA_ID",
    "TRAINING_RECOVERY_FORMAT",
    "ObjectiveConfig",
    "default_objective",
    "ml_contract",
    "objective_config",
    "objective_config_sha256",
    "objective_from_ml_contract",
]
