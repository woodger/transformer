from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.ml import MAX_TARGET_WIDTH
from app.contracts.worker.v8.diagnostics import DiagnosticsConfig

DEFAULT_BATCH_SIZE = 256
DEFAULT_CONTEXT_MODE = "relaxed"
DEFAULT_DETERMINISTIC = False
DEFAULT_DROPOUT = 0.1
DEFAULT_EPOCHS = 25
DEFAULT_HIDDEN = 256
DEFAULT_LAYERS = 5
DEFAULT_LR = 5e-4
DEFAULT_NHEAD = 8
DEFAULT_PATIENCE = 0
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DECAY = 1e-5

DEFAULT_DIRECT_LOSS_WEIGHTS = (1.0,) * MAX_TARGET_WIDTH

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
    out_dim: int = MAX_TARGET_WIDTH
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
        if not 1 <= self.out_dim <= MAX_TARGET_WIDTH:
            raise ValueError(
                f"out_dim must be between 1 and {MAX_TARGET_WIDTH}"
            )
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
            out_dim=_integer(
                values.get("out_dim", MAX_TARGET_WIDTH),
                "out_dim",
            ),
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
    use_amp: bool = False
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    selection: CheckpointSelectionConfig | None = None
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    seed: int = DEFAULT_SEED
    deterministic: bool = DEFAULT_DETERMINISTIC

    def __post_init__(self) -> None:
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be a positive number")
        if isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.epochs, bool) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be a non-negative number")
        raw_selection: object = object.__getattribute__(self, "selection")
        object.__setattr__(
            self,
            "selection",
            _selection_config(raw_selection),
        )
        raw_diagnostics: object = object.__getattribute__(self, "diagnostics")
        object.__setattr__(
            self,
            "diagnostics",
            _diagnostics_config(raw_diagnostics),
        )
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed > 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")

    def to_dict(self) -> JsonObject:
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "use_amp": self.use_amp,
            "weight_decay": self.weight_decay,
            "selection": (
                None if self.selection is None else self.selection.to_dict()
            ),
            "diagnostics": self.diagnostics.to_document(),
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
            "useAmp": "use_amp",
            "weightDecay": "weight_decay",
        }
        for source, target in aliases.items():
            if source in values and target not in values:
                values[target] = values.pop(source)
        allowed = {
            "lr",
            "batch_size",
            "epochs",
            "use_amp",
            "weight_decay",
            "selection",
            "diagnostics",
            "seed",
            "deterministic",
        }
        if set(values) - allowed:
            raise ValueError("training configuration contains unsupported fields")
        return cls(
            lr=_number(values.get("lr", DEFAULT_LR), "lr"),
            batch_size=_integer(
                values.get("batch_size", DEFAULT_BATCH_SIZE),
                "batch_size",
            ),
            epochs=_integer(values.get("epochs", DEFAULT_EPOCHS), "epochs"),
            use_amp=_boolean(values.get("use_amp", False), "use_amp"),
            weight_decay=_number(
                values.get("weight_decay", DEFAULT_WEIGHT_DECAY),
                "weight_decay",
            ),
            selection=CheckpointSelectionConfig.from_dict(
                values.get("selection")
            ),
            diagnostics=DiagnosticsConfig.from_document(
                values.get("diagnostics", DiagnosticsConfig().to_document())
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
        "useAmp": config.use_amp,
        "weightDecay": config.weight_decay,
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


def _selection_config(
    value: object,
) -> CheckpointSelectionConfig | None:
    if isinstance(value, CheckpointSelectionConfig):
        return value
    return CheckpointSelectionConfig.from_dict(value)


def _diagnostics_config(value: object) -> DiagnosticsConfig:
    if isinstance(value, DiagnosticsConfig):
        return value
    return DiagnosticsConfig.from_document(value)
