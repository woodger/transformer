from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.worker.v22.checkpoint_selection import (
    CheckpointSelectionConfig,
)
from app.contracts.worker.v22.diagnostics import DiagnosticsConfig

DEFAULT_BATCH_SIZE = 256
DEFAULT_DETERMINISTIC = False
DEFAULT_EPOCHS = 25
DEFAULT_LR = 5e-4
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DECAY = 1e-5


@dataclass(frozen=True, slots=True)
class TrainConfig:
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

    def to_manifest(self) -> JsonObject:
        return {
            "lr": self.lr,
            "batchSize": self.batch_size,
            "epochs": self.epochs,
            "useAmp": self.use_amp,
            "weightDecay": self.weight_decay,
            "selection": (
                None if self.selection is None else self.selection.to_manifest()
            ),
            "seed": self.seed,
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, data: object) -> TrainConfig | None:
        if not data:
            return None
        values = _object(data)
        aliases = {
            "batchSize": "batch_size",
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
        diagnostics = values.get("diagnostics")
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
            selection=CheckpointSelectionConfig.from_dict(values.get("selection")),
            diagnostics=(
                DiagnosticsConfig()
                if diagnostics is None
                else DiagnosticsConfig.from_document(diagnostics)
            ),
            seed=_integer(values.get("seed", DEFAULT_SEED), "seed"),
            deterministic=_boolean(
                values.get("deterministic", DEFAULT_DETERMINISTIC),
                "deterministic",
            ),
        )


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("training configuration must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError("training configuration field names must be strings")
    return {cast(str, key): item for key, item in mapping.items()}


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DETERMINISTIC",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_SEED",
    "DEFAULT_WEIGHT_DECAY",
    "TrainConfig",
]
