from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.flight.constants import JobState
from app.training.run_config import ModelConfig, TrainConfig


@dataclass(frozen=True, slots=True)
class ExecutionJobRecord:
    """Immutable job projection required by the worker execution boundary."""

    job_id: str
    owner_subject: str
    operation: str
    state: JobState
    selected_device: str | None
    model_label: str | None
    input_model_ref: str | None
    prediction_column: str
    model_config: ModelConfig | None
    training_config: TrainConfig | None
    feature_dim: int | None
    input_frame_count: int
    attempt: int
    queued_at: float | None
    started_at: float | None


@dataclass(frozen=True, slots=True)
class CommittedInputRecord:
    """Durable input fields needed to build a trusted execution plan."""

    job_id: str
    ordinal: int
    rows: int
    byte_count: int
    sha256: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class ModelArtifactRecord:
    """Server-owned checkpoint identity needed by prediction execution."""

    model_ref: str
    owner_subject: str
    checkpoint_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RecoverableAttemptRecord:
    """Process identity required for safe startup recovery."""

    job_id: str
    attempt: int
    pid: int | None
    pgid: int | None
    boot_id: str | None
    process_start_ticks: int | None


def execution_job_from_mapping(
    value: Mapping[str, Any],
) -> ExecutionJobRecord:
    """Compatibility adapter for the public dictionary-based Ledger facade."""

    manifest = value.get("seal_manifest")
    return ExecutionJobRecord(
        job_id=value["job_id"],
        owner_subject=value["owner_subject"],
        operation=value["operation"],
        state=JobState(value["state"]),
        selected_device=value.get("selected_device"),
        model_label=value.get("model_label"),
        input_model_ref=value.get("input_model_ref"),
        prediction_column=value["prediction_column"],
        model_config=ModelConfig.from_dict(value.get("model_config")),
        training_config=TrainConfig.from_dict(value.get("training_config")),
        feature_dim=value.get("feature_dim"),
        input_frame_count=len(manifest or ()),
        attempt=value["attempt"],
        queued_at=value.get("queued_at"),
        started_at=value.get("started_at"),
    )


def recoverable_attempt_from_mapping(
    value: Mapping[str, Any],
) -> RecoverableAttemptRecord:
    """Compatibility adapter for process-recovery callers using mappings."""

    return RecoverableAttemptRecord(
        job_id=value["job_id"],
        attempt=value["attempt"],
        pid=value.get("pid"),
        pgid=value.get("pgid"),
        boot_id=value.get("boot_id"),
        process_start_ticks=value.get("process_start_ticks"),
    )
