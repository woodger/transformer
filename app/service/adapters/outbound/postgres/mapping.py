from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.contracts.worker.v1.config import ModelConfig, TrainConfig
from app.service.adapters.outbound.postgres.models import (
    Job,
    JobInput,
    JobOutput,
    PublishedModel,
)
from app.service.domain.job import JobState
from app.service.domain.records import (
    ExecutionJobRecord,
    InputRecord,
    JobRecord,
    OutputRecord,
    PublishedModelRecord,
    RecoverableAttemptRecord,
)


def execution_job_from_mapping(
    value: Mapping[str, Any],
) -> ExecutionJobRecord:
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
        config_hash=value["config_hash"],
        seal_hash=value.get("seal_hash"),
        feature_dim=value.get("feature_dim"),
        input_frame_count=len(manifest or ()),
        attempt=value["attempt"],
        assigned_device_id=value.get("device_id"),
        resume_generation=value.get("resume_generation"),
        queued_at=value.get("queued_at"),
        started_at=value.get("started_at"),
        attempt_id=value.get("attempt_id"),
    )


def recoverable_attempt_from_mapping(
    value: Mapping[str, Any],
) -> RecoverableAttemptRecord:
    return RecoverableAttemptRecord(
        job_id=value["job_id"],
        attempt=value["attempt"],
        pid=value.get("pid"),
        pgid=value.get("pgid"),
        boot_id=value.get("boot_id"),
        process_start_ticks=value.get("process_start_ticks"),
        attempt_id=value.get("attempt_id"),
    )


def job_record(row: Job | None) -> JobRecord | None:
    if row is None:
        return None
    return JobRecord(
        job_id=row.job_id,
        owner_subject=row.owner_subject,
        operation=row.operation,
        state=JobState(row.state),
        revision=row.revision,
        requested_device=row.requested_device,
        selected_device=row.selected_device,
        prediction_column=row.prediction_column,
        progress=dict(row.progress or {}),
        attempt=row.attempt,
        error_code=row.error_code,
        error_message=row.error_message,
        result=None if row.result is None else dict(row.result),
        created_at=row.created_at.timestamp(),
        updated_at=row.updated_at.timestamp(),
        sealed_at=_timestamp(row.sealed_at),
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
        schema_id=row.schema_id,
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
        metadata_path=row.metadata_path,
        sha256=row.sha256,
        metadata=dict(row.metadata_json),
        producing_job_id=row.producing_job_id,
        created_at=row.created_at.timestamp(),
    )


def _timestamp(value) -> float | None:
    return None if value is None else value.timestamp()


__all__ = [
    "execution_job_from_mapping",
    "input_record",
    "job_record",
    "output_record",
    "published_model_record",
    "recoverable_attempt_from_mapping",
]
