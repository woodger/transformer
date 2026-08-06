from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.flight.v2.constants import FIT_SCHEMA_ID, PREDICT_SCHEMA_ID
from app.service.adapters.outbound.postgres.ledger_support import (
    canonical_uuid,
    decode,
    digest,
    json_value,
    nonnegative,
    now as timestamp_now,
    validate_relative_path,
)
from app.service.adapters.outbound.postgres.models import InputUpload, Job, JobInput
from app.service.domain.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    not_found,
)
from app.service.domain.job import ErrorCode, JobState
from app.service.domain.records import CommittedInputRecord


class InputLedgerSlice:
    """PostgreSQL operations for one job's durable input lifecycle."""

    def __init__(self, sessions):
        self.sessions = sessions
        self.database = sessions.database

    def find_input(
        self,
        job_id: str,
        *,
        ordinal: int | None = None,
        payload_id: str | None = None,
        connection: Session | None = None,
    ):
        if ordinal is None and payload_id is None:
            raise ValueError("ordinal or payload_id is required")
        statement = select(JobInput).where(JobInput.job_id == job_id)
        if ordinal is not None:
            statement = statement.where(JobInput.ordinal == ordinal)
        if payload_id is not None:
            statement = statement.where(JobInput.payload_id == payload_id)
        with self.sessions.read(connection) as session:
            return decode(session.scalar(statement))

    def reserve_input(
        self,
        *,
        job_id: str,
        payload_id: str,
        ordinal: int,
        upload_token: str,
        temporary_path: str,
        storage_class: str = "runtime",
        now: float | None = None,
    ) -> dict:
        nonnegative(ordinal, "ordinal")
        payload_id = canonical_uuid(payload_id, "payload_id")
        validate_relative_path(temporary_path)
        if storage_class not in ("runtime", "recovery"):
            raise ValueError(
                "storage_class must be runtime or recovery"
            )
        reservation = InputUpload(
            upload_token=upload_token,
            job_id=job_id,
            payload_id=payload_id,
            ordinal=ordinal,
            temporary_path=temporary_path,
            storage_class=storage_class,
            created_at=timestamp_now(now),
        )
        try:
            with self.database.transaction() as session:
                job = session.scalar(
                    select(Job)
                    .where(Job.job_id == job_id)
                    .with_for_update()
                )
                if job is None:
                    raise not_found(f"job not found: {job_id}")
                if job.state != JobState.UPLOADING.value:
                    raise failed_precondition("job no longer accepts inputs")
                committed = session.scalar(
                    select(JobInput.job_id).where(
                        JobInput.job_id == job_id,
                        or_(
                            JobInput.ordinal == ordinal,
                            JobInput.payload_id == payload_id,
                        ),
                    )
                )
                if committed is not None:
                    raise conflict(
                        "input ordinal or payloadId is already committed"
                    )
                session.add(reservation)
                session.flush()
        except IntegrityError as exc:
            raise conflict(
                "input ordinal or payloadId is being uploaded"
            ) from exc
        return decode(reservation)

    def abort_input(self, upload_token: str) -> str | None:
        with self.database.transaction() as session:
            upload = session.get(
                InputUpload,
                upload_token,
                with_for_update=True,
            )
            if upload is None:
                return None
            path = upload.temporary_path
            session.delete(upload)
            return path

    def commit_input(
        self,
        *,
        upload_token: str,
        relative_path: str,
        schema_id: str,
        rows: int,
        batches: int,
        byte_count: int,
        sha256: str,
        schema_fingerprint: str,
        source_width: int | None,
        feature_dim: int | None,
        max_payloads: int,
        max_job_bytes: int,
        storage_class: str = "runtime",
        now: float | None = None,
    ) -> dict:
        validate_relative_path(relative_path)
        if storage_class not in ("runtime", "recovery"):
            raise ValueError(
                "storage_class must be runtime or recovery"
            )
        for value, name in (
            (rows, "rows"),
            (batches, "batches"),
            (byte_count, "bytes"),
        ):
            nonnegative(value, name)
        digest(sha256, "sha256")
        digest(schema_fingerprint, "schema_fingerprint")
        timestamp = timestamp_now(now)
        try:
            with self.database.transaction() as session:
                upload = session.get(
                    InputUpload,
                    upload_token,
                    with_for_update=True,
                )
                if upload is None:
                    raise not_found("input upload reservation not found")
                if upload.storage_class != storage_class:
                    raise failed_precondition(
                        "input storage class differs from its reservation"
                    )
                job = session.scalar(
                    select(Job)
                    .where(Job.job_id == upload.job_id)
                    .with_for_update()
                )
                if job.state != JobState.UPLOADING.value:
                    raise failed_precondition("job no longer accepts inputs")
                expected_schema_id = (
                    FIT_SCHEMA_ID
                    if job.operation == "fit"
                    else PREDICT_SCHEMA_ID
                )
                if schema_id != expected_schema_id:
                    raise failed_precondition(
                        f"schemaId {schema_id!r} does not match job operation"
                    )
                payload_count, total_bytes = session.execute(
                    select(
                        func.count(),
                        func.coalesce(func.sum(JobInput.bytes), 0),
                    ).where(JobInput.job_id == upload.job_id)
                ).one()
                if payload_count + 1 > max_payloads:
                    raise ServiceError(
                        ErrorCode.RESOURCE_EXHAUSTED,
                        "job payload quota exceeded",
                    )
                if total_bytes + byte_count > max_job_bytes:
                    raise ServiceError(
                        ErrorCode.RESOURCE_EXHAUSTED,
                        "job byte quota exceeded",
                    )
                existing_contract = session.scalar(
                    select(JobInput)
                    .where(JobInput.job_id == upload.job_id)
                    .order_by(JobInput.ordinal)
                    .limit(1)
                )
                if existing_contract is not None and (
                    existing_contract.schema_id != schema_id
                    or existing_contract.schema_fingerprint
                    != schema_fingerprint
                ):
                    raise failed_precondition(
                        "input schema is inconsistent with committed inputs"
                    )
                known_dimensions = session.execute(
                    select(JobInput.source_width, JobInput.feature_dim)
                    .where(
                        JobInput.job_id == upload.job_id,
                        JobInput.source_width.is_not(None),
                    )
                    .distinct()
                ).all()
                if source_width is not None and any(
                    known_width != source_width or known_dim != feature_dim
                    for known_width, known_dim in known_dimensions
                ):
                    raise failed_precondition(
                        "input dimensions are inconsistent with committed inputs"
                    )
                record = JobInput(
                    job_id=upload.job_id,
                    ordinal=upload.ordinal,
                    payload_id=upload.payload_id,
                    schema_id=schema_id,
                    rows=rows,
                    batches=batches,
                    bytes=byte_count,
                    sha256=sha256,
                    schema_fingerprint=schema_fingerprint,
                    relative_path=relative_path,
                    storage_class=storage_class,
                    source_width=source_width,
                    feature_dim=feature_dim,
                    committed_at=timestamp,
                )
                session.add(record)
                session.delete(upload)
                job.revision += 1
                job.updated_at = timestamp
                session.flush()
                result = decode(record)
                result["revision"] = job.revision
                return result
        except IntegrityError as exc:
            raise conflict(
                "input ordinal or payloadId is already committed"
            ) from exc

    def list_inputs(
        self,
        job_id: str,
        *,
        connection: Session | None = None,
    ) -> list[dict]:
        with self.sessions.read(connection) as session:
            rows = session.scalars(
                select(JobInput)
                .where(JobInput.job_id == job_id)
                .order_by(JobInput.ordinal)
            )
            return [decode(row) for row in rows]

    def list_committed_inputs(
        self,
        job_id: str,
        *,
        connection: Session | None = None,
    ) -> list[CommittedInputRecord]:
        with self.sessions.read(connection) as session:
            rows = session.scalars(
                select(JobInput)
                .where(JobInput.job_id == job_id)
                .order_by(JobInput.ordinal)
            )
            return [_committed_input_record(row) for row in rows]

    def seal_job(
        self,
        job_id: str,
        *,
        manifest_hash: str,
        manifest: list[dict],
        source_width: int | None = None,
        feature_dim: int | None = None,
        result: dict | None = None,
        now: float | None = None,
        connection: Session | None = None,
    ) -> tuple[dict, bool]:
        digest(manifest_hash, "manifest_hash")
        timestamp = timestamp_now(now)
        with self.sessions.write(connection) as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id)
                .with_for_update()
            )
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if job.seal_hash is not None:
                if job.seal_hash != manifest_hash:
                    raise conflict(
                        "job was sealed with a different manifest"
                    )
                return decode(job), True
            if job.state != JobState.UPLOADING.value:
                raise failed_precondition(
                    "job cannot be sealed from its current state"
                )
            active = session.scalar(
                select(InputUpload.upload_token)
                .where(InputUpload.job_id == job_id)
                .limit(1)
            )
            if active is not None:
                raise failed_precondition("job has an upload in progress")
            job.state = JobState.SEALED.value
            job.revision += 1
            job.updated_at = timestamp
            job.sealed_at = timestamp
            job.seal_hash = manifest_hash
            job.seal_manifest = json_value(manifest)
            job.seal_result = json_value(result)
            job.source_width = source_width
            job.feature_dim = feature_dim
            session.flush()
            return decode(job), False


def _committed_input_record(record: JobInput) -> CommittedInputRecord:
    return CommittedInputRecord(
        job_id=record.job_id,
        ordinal=record.ordinal,
        schema_id=record.schema_id,
        rows=record.rows,
        byte_count=record.bytes,
        sha256=record.sha256,
        relative_path=record.relative_path,
        storage_class=record.storage_class,
    )


__all__ = ["InputLedgerSlice"]
