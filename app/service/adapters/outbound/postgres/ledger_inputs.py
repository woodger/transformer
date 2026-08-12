from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.service.adapters.outbound.postgres.ledger_support import (
    LedgerSessions,
    RowMapping,
    canonical_uuid,
    decode,
    decode_optional,
    digest,
    nonnegative,
    now as timestamp_now,
    validate_relative_path,
)
from app.service.adapters.outbound.postgres.models import (
    QUEUE_SEQUENCE,
    InputUpload,
    Job,
    JobInput,
)
from app.service.domain.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    not_found,
)
from app.service.domain.input_manifest import manifest_sha256
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import CommittedInputRecord


class InputLedgerSlice:
    """PostgreSQL operations for one job's durable input lifecycle."""

    def __init__(self, sessions: LedgerSessions) -> None:
        self.sessions = sessions
        self.database = sessions.database

    def find_input(
        self,
        job_id: str,
        *,
        ordinal: int | None = None,
        payload_id: str | None = None,
        connection: Session | None = None,
    ) -> RowMapping | None:
        if ordinal is None and payload_id is None:
            raise ValueError("ordinal or payload_id is required")
        statement = select(JobInput).where(JobInput.job_id == job_id)
        if ordinal is not None:
            statement = statement.where(JobInput.ordinal == ordinal)
        if payload_id is not None:
            statement = statement.where(JobInput.payload_id == payload_id)
        with self.sessions.read(connection) as session:
            return decode_optional(session.scalar(statement))

    def reserve_input(
        self,
        *,
        job_id: str,
        payload_id: str,
        ordinal: int,
        client_execution_id: str,
        fencing_token: int,
        upload_token: str,
        candidate_path: str,
        storage_class: str = "runtime",
        now: float | None = None,
    ) -> RowMapping:
        nonnegative(ordinal, "ordinal")
        payload_id = canonical_uuid(payload_id, "payload_id")
        client_execution_id = canonical_uuid(
            client_execution_id,
            "client_execution_id",
        )
        _positive_fence(fencing_token)
        validate_relative_path(candidate_path)
        _validate_storage_class(storage_class)
        reservation = InputUpload(
            upload_token=upload_token,
            job_id=job_id,
            payload_id=payload_id,
            ordinal=ordinal,
            client_execution_id=client_execution_id,
            fencing_token=fencing_token,
            candidate_path=candidate_path,
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
                _verify_fence(job, client_execution_id, fencing_token)
                if job.input_state != InputState.OPEN.value:
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
            path = upload.candidate_path
            session.delete(upload)
            return path

    def commit_input(
        self,
        *,
        upload_token: str,
        job_id: str,
        client_execution_id: str,
        fencing_token: int,
        relative_path: str,
        schema_id: str,
        data_contract_sha256: str,
        rows: int,
        batches: int,
        byte_count: int,
        sha256: str,
        schema_fingerprint: str,
        source_width: int,
        feature_dim: int,
        selected_device: str,
        max_payloads: int,
        max_job_bytes: int,
        storage_class: str = "runtime",
        now: float | None = None,
    ) -> RowMapping:
        validate_relative_path(relative_path)
        _validate_storage_class(storage_class)
        client_execution_id = canonical_uuid(
            client_execution_id,
            "client_execution_id",
        )
        _positive_fence(fencing_token)
        for value, name in (
            (rows, "rows"),
            (batches, "batches"),
            (byte_count, "bytes"),
            (source_width, "source_width"),
            (feature_dim, "feature_dim"),
        ):
            nonnegative(value, name)
        if source_width == 0 or feature_dim == 0:
            raise ValueError("input dimensions must be positive")
        digest(sha256, "sha256")
        digest(schema_fingerprint, "schema_fingerprint")
        digest(data_contract_sha256, "data_contract_sha256")
        _validate_selected_device(selected_device)
        timestamp = timestamp_now(now)
        try:
            with self.database.transaction() as session:
                upload = session.get(
                    InputUpload,
                    upload_token,
                    with_for_update=True,
                )
                job = session.scalar(
                    select(Job)
                    .where(Job.job_id == job_id)
                    .with_for_update()
                )
                if job is None:
                    raise not_found(f"job not found: {job_id}")
                _verify_fence(job, client_execution_id, fencing_token)
                if upload is None:
                    raise not_found("input upload reservation not found")
                if (
                    upload.job_id != job_id
                    or upload.client_execution_id != client_execution_id
                    or upload.fencing_token != fencing_token
                ):
                    raise _stale_fence()
                if upload.storage_class != storage_class:
                    raise failed_precondition(
                        "input storage class differs from its reservation"
                    )
                if job.input_state != InputState.OPEN.value:
                    raise failed_precondition("job no longer accepts inputs")
                if data_contract_sha256 != job.data_contract_sha256:
                    raise ServiceError(
                        ErrorCode.MODEL_SCHEMA_MISMATCH,
                        "input data contract does not match the job",
                    )
                if (
                    source_width != job.source_width
                    or feature_dim != job.feature_dim
                ):
                    raise failed_precondition(
                        "input dimensions do not match the job data contract"
                    )
                if job.payload_count + 1 > max_payloads:
                    raise ServiceError(
                        ErrorCode.RESOURCE_EXHAUSTED,
                        "job payload quota exceeded",
                    )
                if job.total_bytes + byte_count > max_job_bytes:
                    raise ServiceError(
                        ErrorCode.RESOURCE_EXHAUSTED,
                        "job byte quota exceeded",
                    )
                existing_contract = session.scalar(
                    select(JobInput)
                    .where(JobInput.job_id == job_id)
                    .order_by(JobInput.commit_revision)
                    .limit(1)
                )
                if existing_contract is not None and (
                    existing_contract.schema_id != schema_id
                    or existing_contract.schema_fingerprint
                    != schema_fingerprint
                    or existing_contract.data_contract_sha256
                    != data_contract_sha256
                ):
                    raise failed_precondition(
                        "input schema is inconsistent with committed inputs"
                    )

                prior_next = job.next_input_ordinal
                job.input_revision += 1
                record = JobInput(
                    job_id=job_id,
                    ordinal=upload.ordinal,
                    payload_id=upload.payload_id,
                    commit_revision=job.input_revision,
                    schema_id=schema_id,
                    data_contract_sha256=data_contract_sha256,
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
                session.flush()

                job.next_input_ordinal = _next_input_ordinal(
                    session,
                    job_id,
                    prior_next,
                )
                job.payload_count += 1
                job.total_rows += rows
                job.total_bytes += byte_count
                frontier_advanced = job.next_input_ordinal > prior_next
                if frontier_advanced and job.waiting_for_input:
                    _clear_input_wait(job)

                queued = False
                if job.execution_state == ExecutionState.WAITING_INPUT.value:
                    contiguous_rows = int(session.scalar(
                        select(func.coalesce(func.sum(JobInput.rows), 0)).where(
                            JobInput.job_id == job_id,
                            JobInput.ordinal < job.next_input_ordinal,
                        )
                    ) or 0)
                    if contiguous_rows > 0:
                        _queue_job(job, selected_device, session, timestamp)
                        queued = True

                job.revision += 1
                job.updated_at = timestamp
                session.flush()
                result = decode(record)
                result.update({
                    "revision": job.revision,
                    "input_revision": job.input_revision,
                    "next_input_ordinal": job.next_input_ordinal,
                    "queued": queued,
                    "frontier_advanced": frontier_advanced,
                })
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
    ) -> list[RowMapping]:
        with self.sessions.read(connection) as session:
            rows = session.scalars(
                select(JobInput)
                .where(JobInput.job_id == job_id)
                .order_by(JobInput.ordinal)
            )
            return [decode(row) for row in rows]

    def list_inputs_page(
        self,
        job_id: str,
        owner_subject: str,
        *,
        after_revision: int,
        snapshot_revision: int | None,
        cursor: int | None,
        limit: int,
    ) -> RowMapping:
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(
                    Job.job_id == job_id,
                    Job.owner_subject == owner_subject,
                )
                .with_for_update(read=True)
            )
            if job is None:
                raise not_found("job not found")
            if snapshot_revision is None:
                snapshot_revision = job.input_revision
                cursor = after_revision
            if cursor is None:
                raise failed_precondition(
                    "input pagination cursor is required"
                )
            if snapshot_revision > job.input_revision:
                raise failed_precondition(
                    "snapshotRevision is newer than the current input revision"
                )
            if not after_revision <= cursor <= snapshot_revision:
                raise failed_precondition(
                    "invalid input pagination cursor"
                )
            rows = list(session.scalars(
                select(JobInput)
                .where(
                    JobInput.job_id == job_id,
                    JobInput.commit_revision > cursor,
                    JobInput.commit_revision <= snapshot_revision,
                )
                .order_by(JobInput.commit_revision)
                .limit(limit + 1)
            ))
        has_more = len(rows) > limit
        page = rows[:limit]
        next_cursor = page[-1].commit_revision if has_more else None
        return {
            "after_revision": after_revision,
            "snapshot_revision": snapshot_revision,
            "cursor": cursor,
            "items": [decode(row) for row in page],
            "next_cursor": next_cursor,
            "has_more": has_more,
        }

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

    def close_input(
        self,
        job_id: str,
        *,
        client_execution_id: str,
        fencing_token: int,
        payload_count: int,
        total_rows: int,
        total_bytes: int,
        expected_manifest_sha256: str,
        selected_device: str,
        now: float | None = None,
        connection: Session | None = None,
    ) -> tuple[RowMapping, bool]:
        client_execution_id = canonical_uuid(
            client_execution_id,
            "client_execution_id",
        )
        _positive_fence(fencing_token)
        for value, name in (
            (payload_count, "payload_count"),
            (total_rows, "total_rows"),
            (total_bytes, "total_bytes"),
        ):
            nonnegative(value, name)
        digest(expected_manifest_sha256, "manifest_sha256")
        _validate_selected_device(selected_device)
        timestamp = timestamp_now(now)
        with self.sessions.write(connection) as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id)
                .with_for_update()
            )
            if job is None:
                raise not_found(f"job not found: {job_id}")
            _verify_fence(job, client_execution_id, fencing_token)
            if job.input_state == InputState.CLOSED.value:
                exact = (
                    job.payload_count == payload_count
                    and job.total_rows == total_rows
                    and job.total_bytes == total_bytes
                    and job.manifest_sha256 == expected_manifest_sha256
                )
                if not exact:
                    raise conflict("input was closed with a different manifest")
                return decode(job), True
            if job.input_state != InputState.OPEN.value:
                raise failed_precondition("job input cannot be closed")
            active = session.scalar(
                select(InputUpload.upload_token)
                .where(InputUpload.job_id == job_id)
                .limit(1)
            )
            if active is not None:
                raise failed_precondition("job has an upload in progress")
            rows = list(session.scalars(
                select(JobInput)
                .where(JobInput.job_id == job_id)
                .order_by(JobInput.ordinal)
            ))
            if [row.ordinal for row in rows] != list(range(payload_count)):
                raise failed_precondition(
                    "input ordinals must be contiguous from zero"
                )
            actual_rows = sum(row.rows for row in rows)
            actual_bytes = sum(row.bytes for row in rows)
            if (
                len(rows) != payload_count
                or actual_rows != total_rows
                or actual_bytes != total_bytes
            ):
                raise failed_precondition(
                    "input close totals do not match committed inputs"
                )
            actual_manifest_sha256 = manifest_sha256(rows)
            if actual_manifest_sha256 != expected_manifest_sha256:
                raise failed_precondition(
                    "manifestSha256 does not match committed inputs"
                )
            if job.operation == "fit" and total_rows == 0:
                raise ServiceError(
                    ErrorCode.EMPTY_INPUT,
                    "fit requires at least one input row",
                )

            job.input_state = InputState.CLOSED.value
            job.input_closed_at = timestamp
            job.manifest_sha256 = actual_manifest_sha256
            job.payload_count = payload_count
            job.total_rows = total_rows
            job.total_bytes = total_bytes
            _clear_input_wait(job)
            if job.execution_state == ExecutionState.WAITING_INPUT.value:
                _queue_job(job, selected_device, session, timestamp)
            job.revision += 1
            job.updated_at = timestamp
            session.flush()
            return decode(job), False


def _next_input_ordinal(
    session: Session,
    job_id: str,
    start: int,
) -> int:
    expected = start
    ordinals = session.scalars(
        select(JobInput.ordinal)
        .where(JobInput.job_id == job_id, JobInput.ordinal >= start)
        .order_by(JobInput.ordinal)
    )
    for ordinal in ordinals:
        if ordinal != expected:
            break
        expected += 1
    return expected


def _queue_job(
    job: Job,
    selected_device: str,
    session: Session,
    timestamp: datetime,
) -> None:
    job.execution_state = ExecutionState.QUEUED.value
    job.selected_device = selected_device
    job.queued_at = timestamp
    queue_sequence = session.scalar(QUEUE_SEQUENCE.next_value())
    if queue_sequence is None:
        raise RuntimeError("queue sequence did not return a value")
    job.queue_sequence = queue_sequence


def _clear_input_wait(job: Job) -> None:
    job.waiting_for_input = False
    job.waiting_input_ordinal = None
    job.input_waiting_since = None


def _verify_fence(
    job: Job,
    client_execution_id: str,
    fencing_token: int,
) -> None:
    if (
        job.client_execution_id != client_execution_id
        or job.fencing_token != fencing_token
    ):
        raise _stale_fence()


def _stale_fence() -> ServiceError:
    return ServiceError(
        ErrorCode.STALE_FENCE,
        "job ownership fence is stale",
    )


def _validate_storage_class(value: str) -> None:
    if value not in ("runtime", "recovery"):
        raise ValueError("storage_class must be runtime or recovery")


def _validate_selected_device(value: str) -> None:
    if value not in ("cpu", "cuda"):
        raise ValueError("selected_device must be cpu or cuda")


def _positive_fence(value: int) -> None:
    raw_value = cast(object, value)
    if (
        isinstance(raw_value, bool)
        or not isinstance(raw_value, int)
        or raw_value <= 0
    ):
        raise ValueError("fencing_token must be a positive integer")


def _committed_input_record(record: JobInput) -> CommittedInputRecord:
    return CommittedInputRecord(
        job_id=record.job_id,
        ordinal=record.ordinal,
        payload_id=record.payload_id,
        commit_revision=record.commit_revision,
        schema_id=record.schema_id,
        data_contract_sha256=record.data_contract_sha256,
        rows=record.rows,
        batches=record.batches,
        byte_count=record.bytes,
        sha256=record.sha256,
        schema_fingerprint=record.schema_fingerprint,
        relative_path=record.relative_path,
        storage_class=record.storage_class,
    )


__all__ = ["InputLedgerSlice"]
