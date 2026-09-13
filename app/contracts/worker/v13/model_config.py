from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.semantic.v2.constants import (
    MODEL_ARCHITECTURE_IDENTITY,
    MODEL_ARCHITECTURE_REVISION,
)

DEFAULT_CONTEXT_MODE = "relaxed"
DEFAULT_DROPOUT = 0.1
DEFAULT_HIDDEN = 256
DEFAULT_LAYERS = 5
DEFAULT_NHEAD = 8


@dataclass(frozen=True, slots=True)
class ModelConfig:
    seq_len: int
    feature_dim: int
    hidden: int = DEFAULT_HIDDEN
    layers: int = DEFAULT_LAYERS
    dropout: float = DEFAULT_DROPOUT
    nhead: int = DEFAULT_NHEAD
    context_mode: str = DEFAULT_CONTEXT_MODE
    architecture_identity: str = MODEL_ARCHITECTURE_IDENTITY
    architecture_revision: int = MODEL_ARCHITECTURE_REVISION

    def __post_init__(self) -> None:
        for name, value in (
            ("seq_len", self.seq_len),
            ("feature_dim", self.feature_dim),
            ("hidden", self.hidden),
            ("layers", self.layers),
            ("nhead", self.nhead),
            ("architecture_revision", self.architecture_revision),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden % self.nhead:
            raise ValueError(
                f"hidden ({self.hidden}) must be divisible by nhead ({self.nhead})"
            )
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.context_mode not in ("strict", "relaxed"):
            raise ValueError("context_mode must be one of: strict, relaxed")
        if self.architecture_identity != MODEL_ARCHITECTURE_IDENTITY:
            raise ValueError("model architecture is unavailable")
        if self.architecture_revision != MODEL_ARCHITECTURE_REVISION:
            raise ValueError("model architecture revision is unavailable")

    def to_dict(self) -> JsonObject:
        return {
            "seq_len": self.seq_len,
            "feature_dim": self.feature_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
            "nhead": self.nhead,
            "context_mode": self.context_mode,
            "architecture_identity": self.architecture_identity,
            "architecture_revision": self.architecture_revision,
        }

    def to_manifest(self) -> JsonObject:
        return {
            "architecture": {
                "identity": self.architecture_identity,
                "revision": self.architecture_revision,
            },
            "seqLen": self.seq_len,
            "featureDim": self.feature_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
            "nhead": self.nhead,
            "mode": self.context_mode,
        }

    @classmethod
    def from_dict(cls, data: object) -> ModelConfig | None:
        if not data:
            return None
        values = _object(data)
        if "architecture" in values:
            return cls.from_manifest(values)
        allowed = {
            "seq_len",
            "feature_dim",
            "hidden",
            "layers",
            "dropout",
            "nhead",
            "context_mode",
            "architecture_identity",
            "architecture_revision",
        }
        if set(values) - allowed:
            raise ValueError("model configuration contains unsupported fields")
        return cls(
            seq_len=_integer(values.get("seq_len"), "seq_len"),
            feature_dim=_integer(values.get("feature_dim"), "feature_dim"),
            hidden=_integer(values.get("hidden", DEFAULT_HIDDEN), "hidden"),
            layers=_integer(values.get("layers", DEFAULT_LAYERS), "layers"),
            dropout=_number(values.get("dropout", DEFAULT_DROPOUT), "dropout"),
            nhead=_integer(values.get("nhead", DEFAULT_NHEAD), "nhead"),
            context_mode=_string(
                values.get("context_mode", DEFAULT_CONTEXT_MODE),
                "context_mode",
            ),
            architecture_identity=_string(
                values.get("architecture_identity", MODEL_ARCHITECTURE_IDENTITY),
                "architecture_identity",
            ),
            architecture_revision=_integer(
                values.get("architecture_revision", MODEL_ARCHITECTURE_REVISION),
                "architecture_revision",
            ),
        )

    @classmethod
    def from_manifest(cls, data: object) -> ModelConfig:
        values = _object(data)
        required = {
            "architecture",
            "seqLen",
            "featureDim",
            "hidden",
            "layers",
            "dropout",
            "nhead",
            "mode",
        }
        if set(values) != required:
            raise ValueError("modelConfig has unsupported or missing fields")
        architecture = _object(values["architecture"])
        if set(architecture) != {"identity", "revision"}:
            raise ValueError("modelConfig architecture is invalid")
        return cls(
            seq_len=_integer(values["seqLen"], "seqLen"),
            feature_dim=_integer(values["featureDim"], "featureDim"),
            hidden=_integer(values["hidden"], "hidden"),
            layers=_integer(values["layers"], "layers"),
            dropout=_number(values["dropout"], "dropout"),
            nhead=_integer(values["nhead"], "nhead"),
            context_mode=_string(values["mode"], "mode"),
            architecture_identity=_string(
                architecture["identity"],
                "architecture identity",
            ),
            architecture_revision=_integer(
                architecture["revision"],
                "architecture revision",
            ),
        )


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("model configuration must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError("model configuration field names must be strings")
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


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


__all__ = [
    "DEFAULT_CONTEXT_MODE",
    "DEFAULT_DROPOUT",
    "DEFAULT_HIDDEN",
    "DEFAULT_LAYERS",
    "DEFAULT_NHEAD",
    "ModelConfig",
]
