from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from app.contracts.json_types import JsonObject

DIAGNOSTICS_SCHEMA_VERSION = 3
DEFAULT_GRADIENT_MAX_BATCHES = 1
TARGET_HEAD_FULL_COMMITTED_ARTIFACT = "fullCommittedArtifact"
ENCODER_LAYER_DIRECT_COMPONENT_PER_BATCH = "directComponentPerBatch"


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    gradient_every_steps: int | None = None
    gradient_max_batches: int = DEFAULT_GRADIENT_MAX_BATCHES
    target_head: str | None = None
    encoder_layer_diagnostics: str | None = None

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
        if self.target_head not in {
            None,
            TARGET_HEAD_FULL_COMMITTED_ARTIFACT,
        }:
            raise ValueError("target head diagnostics mode is invalid")
        if self.encoder_layer_diagnostics not in {
            None,
            ENCODER_LAYER_DIRECT_COMPONENT_PER_BATCH,
        }:
            raise ValueError("encoder layer diagnostics mode is invalid")
        if (
            self.encoder_layer_diagnostics is not None
            and self.target_head != TARGET_HEAD_FULL_COMMITTED_ARTIFACT
        ):
            raise ValueError(
                "encoder layer diagnostics require full target head diagnostics"
            )

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
            "targetHead": self.target_head,
            "encoderLayerDiagnostics": self.encoder_layer_diagnostics,
        }

    @classmethod
    def from_document(cls, document: object) -> DiagnosticsConfig:
        if not isinstance(document, Mapping):
            raise ValueError("diagnostics must be an object")
        mapping = cast(Mapping[object, object], document)
        if not all(isinstance(key, str) for key in mapping):
            raise ValueError("diagnostics field names must be strings")
        values = {cast(str, key): value for key, value in mapping.items()}
        if set(values) != {
            "schemaVersion",
            "gradientInteractions",
            "targetHead",
            "encoderLayerDiagnostics",
        }:
            raise ValueError("diagnostics has unsupported or missing fields")
        if values["schemaVersion"] != DIAGNOSTICS_SCHEMA_VERSION:
            raise ValueError(
                f"diagnostics schemaVersion must be {DIAGNOSTICS_SCHEMA_VERSION}"
            )

        interactions = values["gradientInteractions"]
        target_head = values["targetHead"]
        encoder_layer_diagnostics = values["encoderLayerDiagnostics"]
        if target_head not in {
            None,
            TARGET_HEAD_FULL_COMMITTED_ARTIFACT,
        }:
            raise ValueError("targetHead diagnostics mode is invalid")
        if encoder_layer_diagnostics not in {
            None,
            ENCODER_LAYER_DIRECT_COMPONENT_PER_BATCH,
        }:
            raise ValueError("encoderLayerDiagnostics mode is invalid")
        if interactions is None:
            return cls(
                target_head=cast(str | None, target_head),
                encoder_layer_diagnostics=cast(str | None, encoder_layer_diagnostics),
            )
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
        return cls(
            every_steps,
            max_batches,
            cast(str | None, target_head),
            cast(str | None, encoder_layer_diagnostics),
        )


__all__ = [
    "DEFAULT_GRADIENT_MAX_BATCHES",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "ENCODER_LAYER_DIRECT_COMPONENT_PER_BATCH",
    "TARGET_HEAD_FULL_COMMITTED_ARTIFACT",
    "DiagnosticsConfig",
]
