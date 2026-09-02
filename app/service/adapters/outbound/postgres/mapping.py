from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import cast

from app.contracts.json_types import JsonObject
from app.contracts.worker.v10.config import ModelConfig, TrainConfig
from app.service.adapters.outbound.postgres.models import (
    Job,
    JobInput,
    JobOutput,
    PublishedModel,
)
from app.service.domain.job import ExecutionState, InputState
from app.service.domain.records import (
    ExecutionJobRecord,
    InputRecord,
    JobRecord,
    OutputRecord,
    PublishedModelRecord,
    RecoverableAttemptRecord,
)


def execution_job_from_mapping(
    value: Mapping[str, object],
) -> ExecutionJobRecord:
    return ExecutionJobRecord(
        job_id=row_string(value, "job_id"),
        owner_subject=row_string(value, "owner_subject"),
        operation=row_string(value, "operation"),
        input_state=InputState(row_string(value, "input_state")),
        execution_state=ExecutionState(row_string(value, "execution_state")),
        input_revision=row_integer(value, "input_revision"),
        selected_device=row_optional_string(value, "selected_device"),
        model_label=row_optional_string(value, "model_label"),
        input_model_ref=row_optional_string(value, "resolved_model_ref"),
        initialization=row_optional_json_object(value, "initialization"),
        prediction_column=row_string(value, "prediction_column"),
        model_config=ModelConfig.from_dict(value.get("model_config")),
        training_config=TrainConfig.from_dict(value.get("training_config")),
        data_contract=row_json_object(value, "data_contract"),
        ml_contract=row_json_object(value, "ml_contract"),
        config_hash=row_string(value, "config_hash"),
        manifest_sha256=row_optional_string(value, "manifest_sha256"),
        feature_dim=row_integer(value, "feature_dim"),
        input_frame_count=row_integer(value, "next_input_ordinal"),
        attempt=row_integer(value, "attempt"),
        assigned_device_id=row_optional_string(value, "device_id"),
        resume_generation=row_optional_integer(value, "resume_generation"),
        queued_at=row_optional_float(value, "queued_at"),
        started_at=row_optional_float(value, "started_at"),
        attempt_id=row_optional_string(value, "attempt_id"),
    )


def recoverable_attempt_from_mapping(
    value: Mapping[str, object],
) -> RecoverableAttemptRecord:
    return RecoverableAttemptRecord(
        job_id=row_string(value, "job_id"),
        attempt=row_integer(value, "attempt"),
        pid=row_optional_integer(value, "pid"),
        pgid=row_optional_integer(value, "pgid"),
        boot_id=row_optional_string(value, "boot_id"),
        process_start_ticks=row_optional_integer(value, "process_start_ticks"),
        attempt_id=row_optional_string(value, "attempt_id"),
    )


def job_record(row: Job | None) -> JobRecord | None:
    if row is None:
        return None
    return JobRecord(
        job_id=row.job_id,
        owner_subject=row.owner_subject,
        operation=row.operation,
        input_state=InputState(row.input_state),
        execution_state=ExecutionState(row.execution_state),
        revision=row.revision,
        input_revision=row.input_revision,
        next_input_ordinal=row.next_input_ordinal,
        payload_count=row.payload_count,
        total_rows=row.total_rows,
        total_bytes=row.total_bytes,
        manifest_sha256=row.manifest_sha256,
        client_execution_id=row.client_execution_id,
        fencing_token=row.fencing_token,
        requested_device=row.requested_device,
        selected_device=row.selected_device,
        resolved_model_ref=row.resolved_model_ref,
        initialization=(
            None if row.initialization is None else dict(row.initialization)
        ),
        prediction_column=row.prediction_column,
        data_contract=dict(row.data_contract),
        ml_contract=dict(row.ml_contract),
        progress=dict(row.progress or {}),
        attempt=row.attempt,
        error_code=row.error_code,
        error_message=row.error_message,
        result=None if row.result is None else dict(row.result),
        created_at=row.created_at.timestamp(),
        updated_at=row.updated_at.timestamp(),
        input_closed_at=_timestamp(row.input_closed_at),
        queued_at=_timestamp(row.queued_at),
        started_at=_timestamp(row.started_at),
        cancel_requested_at=_timestamp(row.cancel_requested_at),
        finished_at=_timestamp(row.finished_at),
    )


def input_record(row: JobInput) -> InputRecord:
    return InputRecord(
        job_id=row.job_id,
        ordinal=row.ordinal,
        payload_id=row.payload_id,
        commit_revision=row.commit_revision,
        schema_id=row.schema_id,
        data_contract_sha256=row.data_contract_sha256,
        rows=row.rows,
        batches=row.batches,
        byte_count=row.bytes,
        sha256=row.sha256,
        schema_fingerprint=row.schema_fingerprint,
        relative_path=row.relative_path,
        storage_class=row.storage_class,
        source_width=row.source_width,
        feature_dim=row.feature_dim,
        committed_at=row.committed_at.timestamp(),
    )


def output_record(row: JobOutput) -> OutputRecord:
    return OutputRecord(
        job_id=row.job_id,
        ordinal=row.ordinal,
        rows=row.rows,
        batches=row.batches,
        byte_count=row.bytes,
        sha256=row.sha256,
        schema_fingerprint=row.schema_fingerprint,
        relative_path=row.relative_path,
        published_at=row.published_at.timestamp(),
    )


def published_model_record(
    row: PublishedModel | None,
) -> PublishedModelRecord | None:
    if row is None:
        return None
    return PublishedModelRecord(
        model_ref=row.model_ref,
        owner_subject=row.owner_subject,
        label=row.label,
        generation=row.generation,
        checkpoint_path=row.checkpoint_path,
        byte_count=row.checkpoint_bytes,
        sha256=row.sha256,
        metadata=dict(row.metadata_json),
        data_contract=(
            None if row.data_contract is None else dict(row.data_contract)
        ),
        ml_contract=(
            None if row.ml_contract is None else dict(row.ml_contract)
        ),
        objective_config_sha256=row.objective_config_sha256,
        producing_job_id=row.producing_job_id,
        created_at=row.created_at.timestamp(),
    )


def _timestamp(value: datetime | None) -> float | None:
    return None if value is None else value.timestamp()


def row_string(value: Mapping[str, object], key: str) -> str:
    item = _row_value(value, key)
    if not isinstance(item, str):
        raise ValueError(f"database field {key} must be a string")
    return item


def row_optional_string(
    value: Mapping[str, object],
    key: str,
) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        raise ValueError(f"database field {key} must be a string or null")
    return item


def row_integer(value: Mapping[str, object], key: str) -> int:
    item = _row_value(value, key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"database field {key} must be an integer")
    return item


def row_optional_integer(
    value: Mapping[str, object],
    key: str,
) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"database field {key} must be an integer or null")
    return item


def row_boolean(value: Mapping[str, object], key: str) -> bool:
    item = _row_value(value, key)
    if not isinstance(item, bool):
        raise ValueError(f"database field {key} must be a boolean")
    return item


def row_optional_float(
    value: Mapping[str, object],
    key: str,
) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"database field {key} must be numeric or null")
    return float(item)


def row_json_object(
    value: Mapping[str, object],
    key: str,
) -> JsonObject:
    item = _row_value(value, key)
    if not isinstance(item, Mapping):
        raise ValueError(f"database field {key} must be a JSON object")
    mapping = cast(Mapping[object, object], item)
    if not all(isinstance(name, str) for name in mapping):
        raise ValueError(f"database field {key} contains a non-string key")
    return cast(JsonObject, dict(mapping))


def row_optional_json_object(
    value: Mapping[str, object],
    key: str,
) -> JsonObject | None:
    item = value.get(key)
    if item is None:
        return None
    return row_json_object(value, key)


def _row_value(value: Mapping[str, object], key: str) -> object:
    if key not in value:
        raise ValueError(f"database field {key} is missing")
    return value[key]


__all__ = [
    "execution_job_from_mapping",
    "input_record",
    "job_record",
    "output_record",
    "published_model_record",
    "recoverable_attempt_from_mapping",
    "row_boolean",
    "row_integer",
    "row_json_object",
    "row_optional_float",
    "row_optional_integer",
    "row_optional_json_object",
    "row_optional_string",
    "row_string",
]
