from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import cast

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.service.adapters.outbound.postgres.ledger.support import (
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
        chunks: int,
        rows: int,
        native_rows: tuple[int, ...],
        first_range_ordinal: int | None,
        first_example_offset: int | None,
        last_range_ordinal: int | None,
        next_example_offset: int | None,
        batches: int,
        byte_count: int,
        sha256: str,
        schema_fingerprint: str,
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
            (chunks, "chunks"),
            (rows, "rows"),
            (batches, "batches"),
            (byte_count, "bytes"),
        ):
            nonnegative(value, name)
        _validate_receipt_shape(
            chunks=chunks,
            rows=rows,
            native_rows=native_rows,
            first_range_ordinal=first_range_ordinal,
            first_example_offset=first_example_offset,
            last_range_ordinal=last_range_ordinal,
            next_example_offset=next_example_offset,
        )
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
                if len(native_rows) != len(job.total_native_rows):
                    raise failed_precondition(
                        "native row counters do not match sourceEncoding"
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
                    chunks=chunks,
                    rows=rows,
                    native_rows=list(native_rows),
                    first_range_ordinal=first_range_ordinal,
                    first_example_offset=first_example_offset,
                    last_range_ordinal=last_range_ordinal,
                    next_example_offset=next_example_offset,
                    batches=batches,
                    bytes=byte_count,
                    sha256=sha256,
                    schema_fingerprint=schema_fingerprint,
                    relative_path=relative_path,
                    storage_class=storage_class,
                    source_width=job.source_width,
                    feature_dim=job.feature_dim,
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
                job.total_chunks += chunks
                job.total_rows += rows
                job.total_native_rows = [
                    current + added
                    for current, added in zip(
                        job.total_native_rows,
                        native_rows,
                        strict=True,
                    )
                ]
                job.total_bytes += byte_count
                frontier_advanced = job.next_input_ordinal > prior_next
                if frontier_advanced:
                    _validate_contiguous_prefix_boundaries(
                        session,
                        job_id,
                        job.next_input_ordinal,
                    )
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
                    Job.source_encoding.is_not(None),
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
        expected_logical_rows: int | None,
        expected_manifest_sha256: str,
        select_device: Callable[[str, str | None, str, int], str],
        now: float | None = None,
        connection: Session | None = None,
    ) -> tuple[RowMapping, bool]:
        client_execution_id = canonical_uuid(
            client_execution_id,
            "client_execution_id",
        )
        _positive_fence(fencing_token)
        if expected_logical_rows is not None:
            nonnegative(expected_logical_rows, "expected_logical_rows")
        digest(expected_manifest_sha256, "manifest_sha256")
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
                    job.manifest_sha256 == expected_manifest_sha256
                    and (
                        expected_logical_rows is None
                        or job.total_rows == expected_logical_rows
                    )
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
            if [row.ordinal for row in rows] != list(range(len(rows))):
                raise failed_precondition(
                    "input ordinals must be contiguous from zero"
                )
            actual_rows = sum(row.rows for row in rows)
            actual_chunks = sum(row.chunks for row in rows)
            actual_bytes = sum(row.bytes for row in rows)
            actual_native_rows = _sum_native_rows(
                rows,
                width=len(job.total_native_rows),
            )
            if (
                expected_logical_rows is not None
                and actual_rows != expected_logical_rows
            ):
                raise failed_precondition(
                    "expectedLogicalRows does not match committed inputs"
                )
            range_count = _range_count(rows)
            _validate_complete_boundaries(rows, range_count)
            actual_manifest_sha256 = manifest_sha256(rows)
            if actual_manifest_sha256 != expected_manifest_sha256:
                raise failed_precondition(
                    "manifestSha256 does not match committed inputs"
                )
            if job.operation == "fit" and actual_rows == 0:
                raise ServiceError(
                    ErrorCode.EMPTY_INPUT,
                    "fit requires at least one input row",
                )

            selected_device = select_device(
                job.requested_device,
                job.selected_device,
                job.operation,
                actual_rows,
            )
            _validate_selected_device(selected_device)

            job.input_state = InputState.CLOSED.value
            job.input_closed_at = timestamp
            job.manifest_sha256 = actual_manifest_sha256
            job.payload_count = len(rows)
            job.total_chunks = actual_chunks
            job.total_rows = actual_rows
            job.total_native_rows = list(actual_native_rows)
            job.range_count = range_count
            job.total_bytes = actual_bytes
            _clear_input_wait(job)
            if job.execution_state == ExecutionState.WAITING_INPUT.value:
                _queue_job(job, selected_device, session, timestamp)
            job.revision += 1
            job.updated_at = timestamp
            session.flush()
            return decode(job), False


def _validate_receipt_shape(
    *,
    chunks: int,
    rows: int,
    native_rows: tuple[int, ...],
    first_range_ordinal: int | None,
    first_example_offset: int | None,
    last_range_ordinal: int | None,
    next_example_offset: int | None,
) -> None:
    if not native_rows:
        raise ValueError("native_rows must contain one count per feature block")
    for value in native_rows:
        nonnegative(value, "native_rows")
    boundaries = (
        first_range_ordinal,
        first_example_offset,
        last_range_ordinal,
        next_example_offset,
    )
    if chunks == 0:
        if rows != 0 or any(native_rows) or any(
            value is not None for value in boundaries
        ):
            raise ValueError("empty input receipt has inconsistent counters")
        return
    if rows == 0 or any(value is None for value in boundaries):
        raise ValueError("non-empty input receipt has incomplete boundaries")
    for value in cast(tuple[int, int, int, int], boundaries):
        nonnegative(value, "input boundary")
    if cast(int, last_range_ordinal) < cast(int, first_range_ordinal):
        raise ValueError("input receipt range boundaries are reversed")


def _validate_contiguous_prefix_boundaries(
    session: Session,
    job_id: str,
    next_input_ordinal: int,
) -> None:
    rows = list(session.scalars(
        select(JobInput)
        .where(
            JobInput.job_id == job_id,
            JobInput.ordinal < next_input_ordinal,
            JobInput.chunks > 0,
        )
        .order_by(JobInput.ordinal)
    ))
    if not rows:
        return
    if (
        rows[0].first_range_ordinal != 0
        or rows[0].first_example_offset != 0
    ):
        raise failed_precondition(
            "first transmitted range must have rangeOrdinal and exampleOffset zero"
        )
    for previous, current in zip(rows, rows[1:], strict=False):
        _validate_adjacent_boundaries(previous, current)


def _validate_complete_boundaries(
    rows: list[JobInput],
    range_count: int,
) -> None:
    nonempty = [row for row in rows if row.chunks > 0]
    if not nonempty:
        if range_count != 0:
            raise failed_precondition(
                "empty input must declare rangeCount zero"
            )
        return
    first = nonempty[0]
    if first.first_range_ordinal != 0 or first.first_example_offset != 0:
        raise failed_precondition(
            "first transmitted range must have rangeOrdinal and exampleOffset zero"
        )
    for previous, current in zip(nonempty, nonempty[1:], strict=False):
        _validate_adjacent_boundaries(previous, current)
    if nonempty[-1].last_range_ordinal != range_count - 1:
        raise failed_precondition(
            "rangeCount does not match the dense transmitted range ordinals"
        )


def _range_count(rows: list[JobInput]) -> int:
    nonempty = [row for row in rows if row.chunks > 0]
    if not nonempty:
        return 0
    last = nonempty[-1].last_range_ordinal
    if last is None:
        raise failed_precondition("input range boundary metadata is incomplete")
    return last + 1


def _validate_adjacent_boundaries(
    previous: JobInput,
    current: JobInput,
) -> None:
    if (
        previous.last_range_ordinal is None
        or previous.next_example_offset is None
        or current.first_range_ordinal is None
        or current.first_example_offset is None
    ):
        raise failed_precondition("input range boundary metadata is incomplete")
    same_range = (
        current.first_range_ordinal == previous.last_range_ordinal
        and current.first_example_offset == previous.next_example_offset
    )
    next_range = (
        current.first_range_ordinal == previous.last_range_ordinal + 1
        and current.first_example_offset == 0
    )
    if not (same_range or next_range):
        raise failed_precondition(
            "input range chunks are not contiguous across payloads"
        )


def _sum_native_rows(
    rows: list[JobInput],
    *,
    width: int,
) -> tuple[int, ...]:
    if width == 0 or any(len(row.native_rows) != width for row in rows):
        raise failed_precondition(
            "native row counters do not match sourceEncoding"
        )
    totals = [0] * width
    for row in rows:
        for index, count in enumerate(row.native_rows):
            totals[index] += count
    return tuple(totals)


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
        chunks=record.chunks,
        rows=record.rows,
        native_rows=tuple(record.native_rows),
        first_range_ordinal=record.first_range_ordinal,
        first_example_offset=record.first_example_offset,
        last_range_ordinal=record.last_range_ordinal,
        next_example_offset=record.next_example_offset,
        batches=record.batches,
        byte_count=record.bytes,
        sha256=record.sha256,
        schema_fingerprint=record.schema_fingerprint,
        relative_path=record.relative_path,
        storage_class=record.storage_class,
    )


__all__ = ["InputLedgerSlice"]
