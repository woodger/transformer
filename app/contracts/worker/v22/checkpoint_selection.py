from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject

DEFAULT_PATIENCE = 0


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
        return {"min_delta": self.min_delta, "patience": self.patience}

    def to_manifest(self) -> JsonObject:
        return {"minDelta": self.min_delta, "patience": self.patience}

    @classmethod
    def from_dict(cls, data: object) -> CheckpointSelectionConfig | None:
        if data is None:
            return None
        if not isinstance(data, Mapping):
            raise ValueError("selection must be an object")
        mapping = cast(Mapping[object, object], data)
        if not all(isinstance(key, str) for key in mapping):
            raise ValueError("selection field names must be strings")
        values = {cast(str, key): value for key, value in mapping.items()}
        if "minDelta" in values and "min_delta" not in values:
            values["min_delta"] = values.pop("minDelta")
        if set(values) - {"min_delta", "patience"}:
            raise ValueError("selection contains unsupported fields")
        min_delta = values.get("min_delta", 0.0)
        patience = values.get("patience", DEFAULT_PATIENCE)
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)):
            raise ValueError("selection min_delta must be a number")
        if isinstance(patience, bool) or not isinstance(patience, int):
            raise ValueError("selection patience must be an integer")
        return cls(min_delta=float(min_delta), patience=patience)


__all__ = ["DEFAULT_PATIENCE", "CheckpointSelectionConfig"]

