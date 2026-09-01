from __future__ import annotations

from dataclasses import dataclass

from app.contracts.worker.v9.config import ModelConfig
from app.service.domain.job import InputState
from app.service.domain.json_types import JsonObject


@dataclass(frozen=True, slots=True)
class InputUploadMetadata:
    job_id: str
    client_execution_id: str
    fencing_token: int
    payload_id: str
    ordinal: int
    schema_id: str
    input_kind: str
    data_contract_sha256: str
    rows: int


@dataclass(frozen=True, slots=True)
class InputUploadJob:
    job_id: str
    owner_subject: str
    operation: str
    input_state: InputState
    requested_device: str
    selected_device: str | None
    client_execution_id: str
    fencing_token: int
    input_revision: int
    next_input_ordinal: int
    data_contract_sha256: str
    ml_contract: JsonObject
    model_config: ModelConfig


@dataclass(frozen=True, slots=True)
class CommittedInput:
    job_id: str
    payload_id: str
    ordinal: int
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
    input_revision: int = 0
    next_input_ordinal: int = 0
    queued: bool = False
    frontier_advanced: bool = False


@dataclass(frozen=True, slots=True)
class InputUploadAuthorization:
    job: InputUploadJob
    metadata: InputUploadMetadata
    selected_device: str
    storage_class: str
    upload_token: str
    existing: CommittedInput | None


@dataclass(frozen=True, slots=True)
class InputPayloadReceipt:
    relative_path: str
    rows: int
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str
    source_width: int
    feature_dim: int


__all__ = [
    "CommittedInput",
    "InputPayloadReceipt",
    "InputUploadAuthorization",
    "InputUploadJob",
    "InputUploadMetadata",
]
