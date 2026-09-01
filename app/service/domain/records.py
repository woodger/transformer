from __future__ import annotations

from dataclasses import dataclass

from app.contracts.worker.v9.config import ModelConfig, TrainConfig
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.json_types import JsonObject
from app.service.domain.model import ModelLifecycleState


@dataclass(frozen=True, slots=True)
class ExecutionJobRecord:
    """Immutable job projection required by execution orchestration."""

    job_id: str
    owner_subject: str
    operation: str
    input_state: InputState
    execution_state: ExecutionState
    input_revision: int
    selected_device: str | None
    model_label: str | None
    input_model_ref: str | None
    prediction_column: str
    model_config: ModelConfig | None
    training_config: TrainConfig | None
    data_contract: JsonObject
    ml_contract: JsonObject
    config_hash: str
    manifest_sha256: str | None
    feature_dim: int
    input_frame_count: int
    attempt: int
    assigned_device_id: str | None
    resume_generation: int | None
    queued_at: float | None
    started_at: float | None
    initialization: JsonObject | None = None
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class CommittedInputRecord:
    job_id: str
    ordinal: int
    payload_id: str
    commit_revision: int
    schema_id: str
    data_contract_sha256: str
    rows: int
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str
    relative_path: str
    storage_class: str


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    owner_subject: str
    operation: str
    input_state: InputState
    execution_state: ExecutionState
    revision: int
    input_revision: int
    next_input_ordinal: int
    payload_count: int
    total_rows: int
    total_bytes: int
    manifest_sha256: str | None
    client_execution_id: str
    fencing_token: int
    requested_device: str
    selected_device: str | None
    resolved_model_ref: str | None
    prediction_column: str
    data_contract: JsonObject
    ml_contract: JsonObject
    progress: JsonObject
    attempt: int
    error_code: str | None
    error_message: str | None
    result: JsonObject | None
    created_at: float
    updated_at: float
    input_closed_at: float | None
    queued_at: float | None
    started_at: float | None
    cancel_requested_at: float | None
    finished_at: float | None
    initialization: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class InputRecord:
    job_id: str
    ordinal: int
    payload_id: str
    commit_revision: int
    schema_id: str
    data_contract_sha256: str
    rows: int
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str
    relative_path: str
    storage_class: str
    source_width: int
    feature_dim: int
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

    def ledger_record(self) -> JsonObject:
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
    output_count: int
    recovery: StatusRecoveryRecord | None


@dataclass(frozen=True, slots=True)
class ModelArtifactRecord:
    model_ref: str
    owner_subject: str
    checkpoint_path: str
    byte_count: int
    sha256: str
    data_contract: JsonObject | None
    ml_contract: JsonObject | None
    objective_config_sha256: str | None


@dataclass(frozen=True, slots=True)
class PublishedModelRecord:
    model_ref: str
    owner_subject: str
    label: str
    generation: int
    checkpoint_path: str
    metadata_path: str
    byte_count: int
    sha256: str
    metadata: JsonObject
    data_contract: JsonObject | None
    ml_contract: JsonObject | None
    objective_config_sha256: str | None
    producing_job_id: str | None
    created_at: float


@dataclass(frozen=True, slots=True)
class ModelLifecycleRecord:
    model_ref: str
    owner_subject: str
    label: str
    generation: int
    state: ModelLifecycleState
    created_at: float
    deletion_requested_at: float | None
    deleted_at: float | None


@dataclass(frozen=True, slots=True)
class RecoverableAttemptRecord:
    job_id: str
    attempt: int
    pid: int | None
    pgid: int | None
    boot_id: str | None
    process_start_ticks: int | None
    attempt_id: str | None = None
