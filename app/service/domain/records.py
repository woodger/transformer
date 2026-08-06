from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.contracts.worker.v1.config import ModelConfig, TrainConfig
from app.service.domain.job import JobState


@dataclass(frozen=True, slots=True)
class ExecutionJobRecord:
    """Immutable job projection required by execution orchestration."""

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
    config_hash: str
    seal_hash: str | None
    feature_dim: int | None
    input_frame_count: int
    attempt: int
    assigned_device_id: str | None
    resume_generation: int | None
    queued_at: float | None
    started_at: float | None
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class CommittedInputRecord:
    job_id: str
    ordinal: int
    schema_id: str
    rows: int
    byte_count: int
    sha256: str
    relative_path: str
    storage_class: str


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    owner_subject: str
    operation: str
    state: JobState
    revision: int
    requested_device: str
    selected_device: str | None
    prediction_column: str
    progress: dict[str, Any]
    attempt: int
    error_code: str | None
    error_message: str | None
    result: dict[str, Any] | None
    created_at: float
    updated_at: float
    sealed_at: float | None
    queued_at: float | None
    started_at: float | None
    cancel_requested_at: float | None
    finished_at: float | None


@dataclass(frozen=True, slots=True)
class InputRecord:
    job_id: str
    ordinal: int
    payload_id: str
    schema_id: str
    rows: int
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str
    relative_path: str
    storage_class: str
    source_width: int | None
    feature_dim: int | None
    committed_at: float


@dataclass(frozen=True, slots=True)
class OutputRecord:
    job_id: str
    ordinal: int
    rows: int
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str
    relative_path: str
    published_at: float


@dataclass(frozen=True, slots=True)
class StagedPredictionOutput:
    ordinal: int
    rows: int
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str
    relative_path: str

    def ledger_record(self) -> dict:
        return {
            "ordinal": self.ordinal,
            "rows": self.rows,
            "batches": self.batches,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "schema_fingerprint": self.schema_fingerprint,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True, slots=True)
class TrainingRecoveryCheckpointRecord:
    job_id: str
    generation: int
    attempt: int
    format: str
    relative_path: str
    byte_count: int
    sha256: str
    completed_epochs: int
    global_step: int
    training_complete: bool


@dataclass(frozen=True, slots=True)
class StatusRecoveryRecord:
    checkpoint: TrainingRecoveryCheckpointRecord | None
    retry_count: int
    last_retry_code: str | None
    resumed_from_generation: int | None


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    job: JobRecord | None
    inputs: tuple[InputRecord, ...]
    outputs: tuple[OutputRecord, ...]
    recovery: StatusRecoveryRecord | None


@dataclass(frozen=True, slots=True)
class ModelArtifactRecord:
    model_ref: str
    owner_subject: str
    checkpoint_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PublishedModelRecord:
    model_ref: str
    owner_subject: str
    label: str
    generation: int
    checkpoint_path: str
    metadata_path: str
    sha256: str
    metadata: dict[str, Any]
    producing_job_id: str | None
    created_at: float


@dataclass(frozen=True, slots=True)
class RecoverableAttemptRecord:
    job_id: str
    attempt: int
    pid: int | None
    pgid: int | None
    boot_id: str | None
    process_start_ticks: int | None
    attempt_id: str | None = None
