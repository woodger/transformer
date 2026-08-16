from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject

DEFAULT_BATCH_SIZE = 256
DEFAULT_CONTEXT_MODE = "relaxed"
DEFAULT_DETERMINISTIC = False
DEFAULT_DROPOUT = 0.1
DEFAULT_EPOCHS = 25
DEFAULT_HIDDEN = 256
DEFAULT_LAYERS = 5
DEFAULT_LOSS_SCHEDULE = "epoch"
DEFAULT_LOSS_STAGE = 4
DEFAULT_LR = 5e-4
DEFAULT_NHEAD = 8
DEFAULT_PATIENCE = 0
DEFAULT_SEED = 42
DEFAULT_STAGE_SIZE = 5
DEFAULT_WEIGHT_DECAY = 1e-5

DEFAULT_DIRECT_LOSS_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

LOSS_SCHEDULES = ("none", "epoch", "step")

ObjectMap = dict[str, object]


@dataclass(frozen=True, slots=True)
class CheckpointSelectionConfig:
    min_delta: float = 0.0
    patience: int = DEFAULT_PATIENCE

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("selection min_delta must be a non-negative number")
        if isinstance(self.patience, bool) or self.patience < 0:
            raise ValueError("selection patience must be a non-negative integer")
        object.__setattr__(self, "min_delta", float(self.min_delta))

    def to_dict(self) -> JsonObject:
        return {
            "min_delta": self.min_delta,
            "patience": self.patience,
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
    ) -> CheckpointSelectionConfig | None:
        if data is None:
            return None
        values = _object_dict(data, "selection")
        if "minDelta" in values and "min_delta" not in values:
            values["min_delta"] = values.pop("minDelta")
        if set(values) - {"min_delta", "patience"}:
            raise ValueError("selection contains unsupported fields")
        return cls(
            min_delta=_number(values.get("min_delta", 0.0), "selection min_delta"),
            patience=_integer(values.get("patience", DEFAULT_PATIENCE), "selection patience"),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Validated model-shape contract shared by manifests and checkpoints."""

    seq_len: int
    hidden: int = DEFAULT_HIDDEN
    layers: int = DEFAULT_LAYERS
    dropout: float = DEFAULT_DROPOUT
    nhead: int = DEFAULT_NHEAD
    context_mode: str = DEFAULT_CONTEXT_MODE
    out_dim: int = 6
    feature_dim: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seq_len, bool) or self.seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if isinstance(self.hidden, bool) or self.hidden <= 0:
            raise ValueError("hidden must be a positive integer")
        if isinstance(self.layers, bool) or self.layers <= 0:
            raise ValueError("layers must be a positive integer")
        if isinstance(self.nhead, bool) or self.nhead <= 0:
            raise ValueError("nhead must be a positive integer")
        if self.hidden % self.nhead != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be divisible by nhead ({self.nhead})"
            )
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.context_mode not in ("strict", "relaxed"):
            raise ValueError("context_mode must be one of: strict, relaxed")
        if self.out_dim != 6:
            raise ValueError("out_dim must be 6")
        if self.feature_dim is not None and self.feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")

    def to_dict(self) -> JsonObject:
        return {
            "seq_len": self.seq_len,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
            "nhead": self.nhead,
            "context_mode": self.context_mode,
            "out_dim": self.out_dim,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, data: object) -> ModelConfig | None:
        if not data:
            return None
        values = _object_dict(data, "model configuration")
        aliases = {
            "contextMode": "context_mode",
            "featureDim": "feature_dim",
            "hiddenDim": "hidden",
            "hidden_dim": "hidden",
            "mode": "context_mode",
            "numLayers": "layers",
            "num_layers": "layers",
            "outDim": "out_dim",
            "seqLen": "seq_len",
        }
        for source, target in aliases.items():
            if source in values and target not in values:
                values[target] = values.pop(source)
        return cls(
            seq_len=_integer(values.get("seq_len"), "seq_len"),
            hidden=_integer(values.get("hidden", DEFAULT_HIDDEN), "hidden"),
            layers=_integer(values.get("layers", DEFAULT_LAYERS), "layers"),
            dropout=_number(values.get("dropout", DEFAULT_DROPOUT), "dropout"),
            nhead=_integer(values.get("nhead", DEFAULT_NHEAD), "nhead"),
            context_mode=_string(
                values.get("context_mode", DEFAULT_CONTEXT_MODE),
                "context_mode",
            ),
            out_dim=_integer(values.get("out_dim", 6), "out_dim"),
            feature_dim=_optional_integer(
                values.get("feature_dim"),
                "feature_dim",
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Validated optimization contract shared by manifests and checkpoints."""

    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    loss_stage: int = DEFAULT_LOSS_STAGE
    loss_schedule: str = DEFAULT_LOSS_SCHEDULE
    stage_size: int = DEFAULT_STAGE_SIZE
    use_amp: bool = False
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    direct_loss_weights: tuple[float, ...] = DEFAULT_DIRECT_LOSS_WEIGHTS
    selection: CheckpointSelectionConfig | None = None
    seed: int = DEFAULT_SEED
    deterministic: bool = DEFAULT_DETERMINISTIC

    def __post_init__(self) -> None:
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be a positive number")
        if isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.epochs, bool) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.loss_stage != 4:
            raise ValueError("loss_stage must be 4 for the target-aligned objective")
        if self.loss_schedule not in LOSS_SCHEDULES:
            raise ValueError(
                "loss_schedule must be one of: " + ", ".join(LOSS_SCHEDULES)
            )
        if isinstance(self.stage_size, bool) or self.stage_size <= 0:
            raise ValueError("stage_size must be a positive integer")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be a non-negative number")
        # Python callers can bypass annotations, so constructor values are
        # validated before the typed immutable fields become trusted.
        raw_weights: object = object.__getattribute__(
            self,
            "direct_loss_weights",
        )
        weights = _number_tuple(raw_weights, "direct_loss_weights")
        if len(weights) != 6 or any(
            not math.isfinite(value) or value <= 0
            for value in weights
        ):
            raise ValueError("direct_loss_weights must contain six positive numbers")
        object.__setattr__(
            self,
            "direct_loss_weights",
            tuple(float(value) for value in weights),
        )
        raw_selection: object = object.__getattribute__(self, "selection")
        object.__setattr__(
            self,
            "selection",
            _selection_config(raw_selection),
        )
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed > 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")

    def to_dict(self) -> JsonObject:
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "loss_stage": self.loss_stage,
            "loss_schedule": self.loss_schedule,
            "stage_size": self.stage_size,
            "use_amp": self.use_amp,
            "weight_decay": self.weight_decay,
            "direct_loss_weights": list(self.direct_loss_weights),
            "selection": (
                None if self.selection is None else self.selection.to_dict()
            ),
            "seed": self.seed,
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, data: object) -> TrainConfig | None:
        if not data:
            return None
        values = _object_dict(data, "training configuration")
        aliases = {
            "batchSize": "batch_size",
            "deterministic": "deterministic",
            "lossSchedule": "loss_schedule",
            "lossStage": "loss_stage",
            "directLossWeights": "direct_loss_weights",
            "stageSize": "stage_size",
            "useAmp": "use_amp",
            "weightDecay": "weight_decay",
        }
        for source, target in aliases.items():
            if source in values and target not in values:
                values[target] = values.pop(source)
        direct_loss_weights = _number_tuple(
            values.get("direct_loss_weights", DEFAULT_DIRECT_LOSS_WEIGHTS),
            "direct_loss_weights",
        )
        return cls(
            lr=_number(values.get("lr", DEFAULT_LR), "lr"),
            batch_size=_integer(
                values.get("batch_size", DEFAULT_BATCH_SIZE),
                "batch_size",
            ),
            epochs=_integer(values.get("epochs", DEFAULT_EPOCHS), "epochs"),
            loss_stage=_integer(
                values.get("loss_stage", DEFAULT_LOSS_STAGE),
                "loss_stage",
            ),
            loss_schedule=_string(
                values.get("loss_schedule", DEFAULT_LOSS_SCHEDULE),
                "loss_schedule",
            ),
            stage_size=_integer(
                values.get("stage_size", DEFAULT_STAGE_SIZE),
                "stage_size",
            ),
            use_amp=_boolean(values.get("use_amp", False), "use_amp"),
            weight_decay=_number(
                values.get("weight_decay", DEFAULT_WEIGHT_DECAY),
                "weight_decay",
            ),
            direct_loss_weights=direct_loss_weights,
            selection=CheckpointSelectionConfig.from_dict(
                values.get("selection")
            ),
            seed=_integer(values.get("seed", DEFAULT_SEED), "seed"),
            deterministic=_boolean(
                values.get("deterministic", DEFAULT_DETERMINISTIC),
                "deterministic",
            ),
        )


def model_config_to_manifest(config: ModelConfig) -> JsonObject:
    return {
        "seqLen": config.seq_len,
        "hidden": config.hidden,
        "layers": config.layers,
        "dropout": config.dropout,
        "nhead": config.nhead,
        "mode": config.context_mode,
        "outDim": config.out_dim,
        "featureDim": config.feature_dim,
    }


def train_config_to_manifest(config: TrainConfig) -> JsonObject:
    return {
        "lr": config.lr,
        "batchSize": config.batch_size,
        "epochs": config.epochs,
        "lossStage": config.loss_stage,
        "lossSchedule": config.loss_schedule,
        "stageSize": config.stage_size,
        "useAmp": config.use_amp,
        "weightDecay": config.weight_decay,
        "directLossWeights": list(config.direct_loss_weights),
        "selection": (
            None
            if config.selection is None
            else {
                "minDelta": config.selection.min_delta,
                "patience": config.selection.patience,
            }
        ),
        "seed": config.seed,
        "deterministic": config.deterministic,
    }


def _object_dict(value: object, label: str) -> ObjectMap:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{label} field names must be strings")
    return {cast(str, key): item for key, item in mapping.items()}


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _number_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    items = cast(Sequence[object], value)
    return tuple(_number(item, label) for item in items)


def _selection_config(
    value: object,
) -> CheckpointSelectionConfig | None:
    if isinstance(value, CheckpointSelectionConfig):
        return value
    return CheckpointSelectionConfig.from_dict(value)
