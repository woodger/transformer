from __future__ import annotations

from dataclasses import dataclass

from app.contracts.worker.v17.config import ModelConfig
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
    chunks: int
    rows: int
    native_rows: tuple[int, ...]


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
    source_encoding: JsonObject
    model_contract: JsonObject
    model_config: ModelConfig


@dataclass(frozen=True, slots=True)
class CommittedInput:
    job_id: str
    payload_id: str
    ordinal: int
    schema_id: str
    data_contract_sha256: str
    chunks: int
    rows: int
    native_rows: tuple[int, ...]
    first_range_ordinal: int | None
    first_example_offset: int | None
    last_range_ordinal: int | None
    next_example_offset: int | None
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str
    relative_path: str
    storage_class: str
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
    chunks: int
    rows: int
    native_rows: tuple[int, ...]
    first_range_ordinal: int | None
    first_example_offset: int | None
    last_range_ordinal: int | None
    next_example_offset: int | None
    batches: int
    byte_count: int
    sha256: str
    schema_fingerprint: str


__all__ = [
    "CommittedInput",
    "InputPayloadReceipt",
    "InputUploadAuthorization",
    "InputUploadJob",
    "InputUploadMetadata",
]
