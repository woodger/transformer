from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict | None,
    ) -> CheckpointSelectionConfig | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("selection must be an object or null")
        values = dict(data)
        if "minDelta" in values and "min_delta" not in values:
            values["min_delta"] = values.pop("minDelta")
        if set(values) - {"min_delta", "patience"}:
            raise ValueError("selection contains unsupported fields")
        return cls(**values)


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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> ModelConfig | None:
        if not data:
            return None
        values = dict(data)
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
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})


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
        weights = tuple(self.direct_loss_weights)
        if len(weights) != 6 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in weights
        ):
            raise ValueError("direct_loss_weights must contain six positive numbers")
        object.__setattr__(
            self,
            "direct_loss_weights",
            tuple(float(value) for value in weights),
        )
        if self.selection is not None and not isinstance(
            self.selection,
            CheckpointSelectionConfig,
        ):
            object.__setattr__(
                self,
                "selection",
                CheckpointSelectionConfig.from_dict(self.selection),
            )
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed > 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> TrainConfig | None:
        if not data:
            return None
        values = dict(data)
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
        if "selection" in values:
            values["selection"] = CheckpointSelectionConfig.from_dict(
                values["selection"]
            )
        if "direct_loss_weights" in values:
            values["direct_loss_weights"] = tuple(values["direct_loss_weights"])
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})


def model_config_to_manifest(config: ModelConfig) -> dict:
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


def train_config_to_manifest(config: TrainConfig) -> dict:
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
