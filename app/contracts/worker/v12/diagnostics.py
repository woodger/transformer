from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject

DIAGNOSTICS_SCHEMA_VERSION = 1
DEFAULT_GRADIENT_MAX_BATCHES = 1


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    gradient_every_steps: int | None = None
    gradient_max_batches: int = DEFAULT_GRADIENT_MAX_BATCHES

    def __post_init__(self) -> None:
        interval = self.gradient_every_steps
        if interval is not None and (
            isinstance(interval, bool) or interval <= 0
        ):
            raise ValueError("gradient sample interval must be a positive integer")
        if (
            isinstance(self.gradient_max_batches, bool)
            or self.gradient_max_batches <= 0
        ):
            raise ValueError("gradient max batches must be a positive integer")

    @property
    def gradient_sample_every_steps(self) -> int | None:
        return self.gradient_every_steps

    def to_document(self) -> JsonObject:
        return {
            "schemaVersion": DIAGNOSTICS_SCHEMA_VERSION,
            "gradientInteractions": (
                None
                if self.gradient_every_steps is None
                else {
                    "everySteps": self.gradient_every_steps,
                    "maxBatches": self.gradient_max_batches,
                }
            ),
        }

    @classmethod
    def from_document(cls, document: object) -> DiagnosticsConfig:
        if not isinstance(document, Mapping):
            raise ValueError("diagnostics must be an object")
        mapping = cast(Mapping[object, object], document)
        if not all(isinstance(key, str) for key in mapping):
            raise ValueError("diagnostics field names must be strings")
        values = {cast(str, key): value for key, value in mapping.items()}
        if set(values) != {"schemaVersion", "gradientInteractions"}:
            raise ValueError("diagnostics has unsupported or missing fields")
        if values["schemaVersion"] != DIAGNOSTICS_SCHEMA_VERSION:
            raise ValueError(
                f"diagnostics schemaVersion must be {DIAGNOSTICS_SCHEMA_VERSION}"
            )
        interactions = values["gradientInteractions"]
        if interactions is None:
            return cls()
        if not isinstance(interactions, Mapping):
            raise ValueError("gradientInteractions must be an object or null")
        typed = cast(Mapping[object, object], interactions)
        if set(typed) != {"everySteps", "maxBatches"}:
            raise ValueError("gradientInteractions has unsupported or missing fields")
        every_steps = typed["everySteps"]
        max_batches = typed["maxBatches"]
        if isinstance(every_steps, bool) or not isinstance(every_steps, int):
            raise ValueError("everySteps must be a positive integer")
        if isinstance(max_batches, bool) or not isinstance(max_batches, int):
            raise ValueError("maxBatches must be a positive integer")
        return cls(every_steps, max_batches)


__all__ = [
    "DEFAULT_GRADIENT_MAX_BATCHES",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "DiagnosticsConfig",
]

