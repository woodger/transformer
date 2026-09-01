from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject

DIAGNOSTICS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    """Observational training diagnostics excluded from objective identity."""

    gradient_sample_every_steps: int | None = None

    def __post_init__(self) -> None:
        interval = self.gradient_sample_every_steps
        if interval is not None and (
            isinstance(interval, bool) or interval <= 0
        ):
            raise ValueError("gradient sample interval must be a positive integer")

    def to_document(self) -> JsonObject:
        return {
            "schemaVersion": DIAGNOSTICS_SCHEMA_VERSION,
            "gradientInteractions": (
                None
                if self.gradient_sample_every_steps is None
                else {"sampleEverySteps": self.gradient_sample_every_steps}
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
        if set(typed) != {"sampleEverySteps"}:
            raise ValueError("gradientInteractions has unsupported or missing fields")
        interval = typed["sampleEverySteps"]
        if isinstance(interval, bool) or not isinstance(interval, int):
            raise ValueError("sampleEverySteps must be a positive integer")
        return cls(gradient_sample_every_steps=interval)


__all__ = [
    "DIAGNOSTICS_SCHEMA_VERSION",
    "DiagnosticsConfig",
]
