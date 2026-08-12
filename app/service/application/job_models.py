from __future__ import annotations

from dataclasses import dataclass

from app.contracts.worker.v3.config import ModelConfig, TrainConfig
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.json_types import JsonObject
from app.service.domain.records import (
    InputRecord,
    OutputRecord,
    PublishedModelRecord,
    StatusSnapshot,
)


@dataclass(frozen=True, slots=True)
class ServiceLimits:
    max_message_bytes: int
    target_batch_bytes: int
    max_batch_bytes: int
    max_payload_bytes: int
    max_rows_per_payload: int
    max_payloads_per_job: int
    max_job_bytes: int
    max_active_jobs_per_subject: int
    max_page_items: int
    input_idle_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class CreateJobCommand:
    owner_subject: str
    request_id: str
    idempotency_key: str
    request_hash: str
    job_id: str
    client_execution_id: str
    operation: str
    requested_device: str
    prediction_column: str
    data_contract: JsonObject
    ml_contract: JsonObject
    model_label: str | None = None
    model_selector: str | None = None
    model_ref: str | None = None
    model_config: ModelConfig | None = None
    training_config: TrainConfig | None = None


@dataclass(frozen=True, slots=True)
class AcquireJobCommand:
    owner_subject: str
    request_id: str
    idempotency_key: str
    request_hash: str
    job_id: str
    previous_client_execution_id: str
    expected_fencing_token: int
    client_execution_id: str


@dataclass(frozen=True, slots=True)
class CloseInputCommand:
    owner_subject: str
    request_id: str
    idempotency_key: str
    request_hash: str
    job_id: str
    client_execution_id: str
    fencing_token: int
    payload_count: int
    total_rows: int
    total_bytes: int
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CancelJobCommand:
    owner_subject: str
    request_id: str
    idempotency_key: str
    request_hash: str
    job_id: str
    client_execution_id: str
    fencing_token: int


@dataclass(frozen=True, slots=True)
class GetJobStatusQuery:
    owner_subject: str
    request_id: str
    job_id: str


@dataclass(frozen=True, slots=True)
class ListJobInputsQuery:
    owner_subject: str
    request_id: str
    job_id: str
    after_revision: int
    snapshot_revision: int | None
    cursor: int | None
    limit: int


@dataclass(frozen=True, slots=True)
class ListJobOutputsQuery:
    owner_subject: str
    request_id: str
    job_id: str
    cursor: int | None
    limit: int


@dataclass(frozen=True, slots=True)
class DescribeModelQuery:
    owner_subject: str
    request_id: str
    model_selector: str
    model_ref: str


@dataclass(frozen=True, slots=True)
class JobCreated:
    request_id: str
    job_id: str
    operation: str
    revision: int
    input_state: InputState
    input_revision: int
    next_input_ordinal: int
    execution_state: ExecutionState
    client_execution_id: str
    fencing_token: int
    requested_device: str
    selected_device: str | None
    resolved_model_ref: str | None
    data_contract: JsonObject
    ml_contract: JsonObject
    limits: ServiceLimits


@dataclass(frozen=True, slots=True)
class JobAcquired:
    request_id: str
    job_id: str
    revision: int
    client_execution_id: str
    fencing_token: int


@dataclass(frozen=True, slots=True)
class InputClosed:
    request_id: str
    job_id: str
    revision: int
    input_state: InputState
    input_revision: int
    payload_count: int
    total_rows: int
    total_bytes: int
    manifest_sha256: str
    execution_state: ExecutionState


@dataclass(frozen=True, slots=True)
class JobCancelled:
    request_id: str
    job_id: str
    revision: int
    input_state: InputState
    execution_state: ExecutionState


@dataclass(frozen=True, slots=True)
class JobCreationPreparation:
    result: JobCreated
    model_config: ModelConfig
    training_config: TrainConfig | None
    resolved_model_ref: str | None


@dataclass(frozen=True, slots=True)
class StoredInputPage:
    after_revision: int
    snapshot_revision: int
    cursor: int
    items: tuple[InputRecord, ...]
    next_cursor: int | None
    has_more: bool


@dataclass(frozen=True, slots=True)
class StoredOutputPage:
    cursor: int | None
    items: tuple[OutputRecord, ...]
    next_cursor: int | None
    has_more: bool


@dataclass(frozen=True, slots=True)
class JobStatusResult:
    request_id: str
    snapshot: StatusSnapshot


@dataclass(frozen=True, slots=True)
class JobInputsPage:
    request_id: str
    job_id: str
    after_revision: int
    snapshot_revision: int
    cursor: int
    items: tuple[InputRecord, ...]
    next_cursor: int | None
    has_more: bool


@dataclass(frozen=True, slots=True)
class JobOutputsPage:
    request_id: str
    job_id: str
    cursor: int | None
    items: tuple[OutputRecord, ...]
    next_cursor: int | None
    has_more: bool


@dataclass(frozen=True, slots=True)
class ModelDescription:
    request_id: str
    model: PublishedModelRecord
    model_config: ModelConfig


__all__ = [
    "AcquireJobCommand",
    "CancelJobCommand",
    "CloseInputCommand",
    "CreateJobCommand",
    "DescribeModelQuery",
    "GetJobStatusQuery",
    "InputClosed",
    "JobAcquired",
    "JobCancelled",
    "JobCreated",
    "JobCreationPreparation",
    "JobInputsPage",
    "JobOutputsPage",
    "JobStatusResult",
    "ListJobInputsQuery",
    "ListJobOutputsQuery",
    "ModelDescription",
    "ServiceLimits",
    "StoredInputPage",
    "StoredOutputPage",
]
