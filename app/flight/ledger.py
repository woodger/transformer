from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import secrets
import uuid
from typing import Callable, Iterator, Sequence

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.migrations import require_current_schema
from app.database.models import (
    IdempotencyRecord,
    InputUpload,
    Job,
    JobAttempt,
    JobInput,
    JobOutput,
    ModelAlias,
    OutputTicket,
    PublishedModel,
    QUEUE_SEQUENCE,
    RuntimeState,
)
from app.database.session import Database
from app.flight.constants import (
    ErrorCode,
    FIT_SCHEMA_ID,
    JobState,
    PREDICT_SCHEMA_ID,
    SUPPORTED_DEVICES,
    SUPPORTED_OPERATIONS,
)
from app.flight.errors import ServiceError, conflict, failed_precondition, not_found
from app.flight.state import validate_transition


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_STATES = (
    JobState.SUCCEEDED.value,
    JobState.FAILED.value,
    JobState.CANCELLED.value,
)


class Ledger:
    """PostgreSQL source of truth for Flight jobs and published artifacts."""

    def __init__(self, database: Database):
        if not isinstance(database, Database):
            raise TypeError("Ledger requires a PostgreSQL Database")
        self.database = database

    def initialize(self) -> "Ledger":
        require_current_schema(self.database.config)
        return self

    def close(self) -> None:
        self.database.close()

    def healthcheck(self) -> bool:
        with self.database.session() as session:
            session.scalar(select(Job.job_id).limit(1))
        return True

    @contextmanager
    def connection(self) -> Iterator[Session]:
        """Expose a read-only-by-default ORM session for diagnostics."""
        with self.database.session() as session:
            yield session

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self.database.transaction() as session:
            yield session

    @contextmanager
    def _read(self, connection: Session | None):
        if connection is not None:
            yield connection
            return
        with self.database.session() as session:
            yield session

    @contextmanager
    def _write(self, connection: Session | None):
        if connection is not None:
            yield connection
            return
        with self.database.transaction() as session:
            yield session

    def create_job(
        self,
        *,
        job_id: str,
        owner_subject: str,
        operation: str,
        requested_device: str,
        prediction_column: str,
        config_hash: str,
        model_label: str | None = None,
        input_model_ref: str | None = None,
        model_config=None,
        training_config=None,
        now: float | None = None,
        connection: Session | None = None,
    ) -> dict:
        job_id = _canonical_uuid(job_id, "job_id")
        if not owner_subject:
            raise ValueError("owner_subject must not be empty")
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"unsupported operation: {operation}")
        if requested_device not in SUPPORTED_DEVICES:
            raise ValueError(f"unsupported device: {requested_device}")
        _digest(config_hash, "config_hash")
        if operation == "fit" and (not model_label or input_model_ref is not None):
            raise ValueError("fit job requires model_label only")
        if operation == "predict" and (not input_model_ref or model_label is not None):
            raise ValueError("predict job requires input_model_ref only")
        timestamp = _now(now)
        job = Job(
            job_id=job_id,
            owner_subject=owner_subject,
            operation=operation,
            state=JobState.UPLOADING.value,
            revision=1,
            requested_device=requested_device,
            model_label=model_label,
            input_model_ref=input_model_ref,
            prediction_column=prediction_column,
            model_config=_json_value(model_config),
            training_config=_json_value(training_config),
            config_hash=config_hash,
            progress={},
            attempt=0,
            created_at=timestamp,
            updated_at=timestamp,
        )
        try:
            with self._write(connection) as session:
                session.add(job)
                session.flush()
        except IntegrityError as exc:
            raise conflict(f"job already exists: {job_id}") from exc
        return _decode(job)

    def get_job(
        self,
        job_id: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
        for_update: bool = False,
    ) -> dict | None:
        statement = select(Job).where(Job.job_id == job_id)
        if owner_subject is not None:
            statement = statement.where(Job.owner_subject == owner_subject)
        if for_update:
            statement = statement.with_for_update()
        with self._read(connection) as session:
            return _decode(session.scalar(statement))

    def active_job_count(self, owner_subject: str, *, connection: Session | None = None) -> int:
        with self._read(connection) as session:
            return int(session.scalar(
                select(func.count()).select_from(Job).where(
                    Job.owner_subject == owner_subject,
                    Job.state.not_in(_TERMINAL_STATES),
                )
            ) or 0)

    def get_status_snapshot(self, job_id: str, owner_subject: str):
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(Job.job_id == job_id, Job.owner_subject == owner_subject)
                .with_for_update(read=True)
            )
            inputs = session.scalars(
                select(JobInput).where(JobInput.job_id == job_id).order_by(JobInput.ordinal)
            ).all()
            outputs = session.scalars(
                select(JobOutput).where(JobOutput.job_id == job_id).order_by(JobOutput.ordinal)
            ).all()
        return _decode(job), [_decode(row) for row in inputs], [_decode(row) for row in outputs]

    def list_jobs(
        self,
        states: Sequence[str | JobState] | None = None,
        *,
        connection: Session | None = None,
    ) -> list[dict]:
        statement = select(Job)
        if states:
            statement = statement.where(
                Job.state.in_([JobState(value).value for value in states])
            )
        statement = statement.order_by(Job.created_at, Job.job_id)
        with self._read(connection) as session:
            return [_decode(row) for row in session.scalars(statement)]

    def transition_job(
        self,
        job_id: str,
        target_state: str | JobState,
        *,
        expected_revision: int | None = None,
        updates: dict | None = None,
        now: float | None = None,
        connection: Session | None = None,
    ) -> dict:
        target_state = JobState(target_state)
        timestamp = _now(now)
        with self._write(connection) as session:
            job = session.scalar(select(Job).where(Job.job_id == job_id).with_for_update())
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if expected_revision is not None and job.revision != expected_revision:
                raise failed_precondition("job revision has changed")
            validate_transition(job.state, target_state)
            job.state = target_state.value
            job.revision += 1
            job.updated_at = timestamp
            for name, value in _transition_updates(updates or {}).items():
                setattr(job, name, value)
            session.flush()
            return _decode(job)

    def update_progress(self, job_id: str, progress: dict, *, now: float | None = None) -> dict:
        with self.database.transaction() as session:
            job = session.scalar(select(Job).where(Job.job_id == job_id).with_for_update())
            if job is None or job.state not in (
                JobState.RUNNING.value,
                JobState.CANCELLING.value,
            ):
                raise failed_precondition("job is not running")
            job.progress = _json_value(progress)
            job.revision += 1
            job.updated_at = _now(now)
            session.flush()
            return _decode(job)

    def lookup_idempotency(
        self,
        owner_subject: str,
        action_name: str,
        idempotency_key: str,
        *,
        connection: Session | None = None,
    ) -> dict | None:
        with self._read(connection) as session:
            return _decode(session.get(
                IdempotencyRecord,
                (owner_subject, action_name, idempotency_key),
            ))

    def record_idempotency(
        self,
        *,
        owner_subject: str,
        action_name: str,
        idempotency_key: str,
        request_hash: str,
        response: dict,
        job_id: str | None = None,
        now: float | None = None,
        connection: Session | None = None,
    ) -> tuple[dict, bool]:
        with self._write(connection) as session:
            _advisory_lock(session, "idempotency", owner_subject, action_name, idempotency_key)
            existing = session.get(
                IdempotencyRecord,
                (owner_subject, action_name, idempotency_key),
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise conflict("idempotency key was used for a different request")
                return existing.response, True
            session.add(IdempotencyRecord(
                owner_subject=owner_subject,
                action_name=action_name,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                response=_json_value(response),
                job_id=job_id,
                created_at=_now(now),
            ))
            session.flush()
        return response, False

    def run_idempotent(
        self,
        *,
        owner_subject: str,
        action_name: str,
        idempotency_key: str,
        request_hash: str,
        mutation: Callable[[Session], tuple[dict, str | None]],
        now: float | None = None,
    ) -> tuple[dict, bool]:
        with self.database.transaction() as session:
            _advisory_lock(session, "idempotency", owner_subject, action_name, idempotency_key)
            existing = session.get(
                IdempotencyRecord,
                (owner_subject, action_name, idempotency_key),
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise conflict("idempotency key was used for a different request")
                return existing.response, True
            response, job_id = mutation(session)
            session.add(IdempotencyRecord(
                owner_subject=owner_subject,
                action_name=action_name,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                response=_json_value(response),
                job_id=job_id,
                created_at=_now(now),
            ))
            session.flush()
            return response, False

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
        with self._read(connection) as session:
            return _decode(session.scalar(statement))

    def reserve_input(
        self,
        *,
        job_id: str,
        payload_id: str,
        ordinal: int,
        upload_token: str,
        temporary_path: str,
        now: float | None = None,
    ) -> dict:
        _nonnegative(ordinal, "ordinal")
        payload_id = _canonical_uuid(payload_id, "payload_id")
        _validate_relative_path(temporary_path)
        reservation = InputUpload(
            upload_token=upload_token,
            job_id=job_id,
            payload_id=payload_id,
            ordinal=ordinal,
            temporary_path=temporary_path,
            created_at=_now(now),
        )
        try:
            with self.database.transaction() as session:
                job = session.scalar(select(Job).where(Job.job_id == job_id).with_for_update())
                if job is None:
                    raise not_found(f"job not found: {job_id}")
                if job.state != JobState.UPLOADING.value:
                    raise failed_precondition("job no longer accepts inputs")
                committed = session.scalar(select(JobInput.job_id).where(
                    JobInput.job_id == job_id,
                    or_(JobInput.ordinal == ordinal, JobInput.payload_id == payload_id),
                ))
                if committed is not None:
                    raise conflict("input ordinal or payloadId is already committed")
                session.add(reservation)
                session.flush()
        except IntegrityError as exc:
            raise conflict("input ordinal or payloadId is being uploaded") from exc
        return _decode(reservation)

    def abort_input(self, upload_token: str) -> str | None:
        with self.database.transaction() as session:
            upload = session.get(InputUpload, upload_token, with_for_update=True)
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
        now: float | None = None,
    ) -> dict:
        _validate_relative_path(relative_path)
        for value, name in ((rows, "rows"), (batches, "batches"), (byte_count, "bytes")):
            _nonnegative(value, name)
        _digest(sha256, "sha256")
        _digest(schema_fingerprint, "schema_fingerprint")
        timestamp = _now(now)
        try:
            with self.database.transaction() as session:
                upload = session.get(InputUpload, upload_token, with_for_update=True)
                if upload is None:
                    raise not_found("input upload reservation not found")
                job = session.scalar(
                    select(Job).where(Job.job_id == upload.job_id).with_for_update()
                )
                if job.state != JobState.UPLOADING.value:
                    raise failed_precondition("job no longer accepts inputs")
                expected_schema_id = FIT_SCHEMA_ID if job.operation == "fit" else PREDICT_SCHEMA_ID
                if schema_id != expected_schema_id:
                    raise failed_precondition(
                        f"schemaId {schema_id!r} does not match job operation"
                    )
                payload_count, total_bytes = session.execute(
                    select(func.count(), func.coalesce(func.sum(JobInput.bytes), 0)).where(
                        JobInput.job_id == upload.job_id
                    )
                ).one()
                if payload_count + 1 > max_payloads:
                    raise ServiceError(ErrorCode.RESOURCE_EXHAUSTED, "job payload quota exceeded")
                if total_bytes + byte_count > max_job_bytes:
                    raise ServiceError(ErrorCode.RESOURCE_EXHAUSTED, "job byte quota exceeded")
                existing_contract = session.scalar(
                    select(JobInput)
                    .where(JobInput.job_id == upload.job_id)
                    .order_by(JobInput.ordinal)
                    .limit(1)
                )
                if existing_contract is not None and (
                    existing_contract.schema_id != schema_id
                    or existing_contract.schema_fingerprint != schema_fingerprint
                ):
                    raise failed_precondition("input schema is inconsistent with committed inputs")
                known_dimensions = session.execute(
                    select(JobInput.source_width, JobInput.feature_dim)
                    .where(JobInput.job_id == upload.job_id, JobInput.source_width.is_not(None))
                    .distinct()
                ).all()
                if source_width is not None and any(
                    known_width != source_width or known_dim != feature_dim
                    for known_width, known_dim in known_dimensions
                ):
                    raise failed_precondition("input dimensions are inconsistent with committed inputs")
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
                    source_width=source_width,
                    feature_dim=feature_dim,
                    committed_at=timestamp,
                )
                session.add(record)
                session.delete(upload)
                job.revision += 1
                job.updated_at = timestamp
                session.flush()
                result = _decode(record)
                result["revision"] = job.revision
                return result
        except IntegrityError as exc:
            raise conflict("input ordinal or payloadId is already committed") from exc

    def list_inputs(
        self,
        job_id: str,
        *,
        connection: Session | None = None,
    ) -> list[dict]:
        with self._read(connection) as session:
            rows = session.scalars(
                select(JobInput).where(JobInput.job_id == job_id).order_by(JobInput.ordinal)
            )
            return [_decode(row) for row in rows]

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
        _digest(manifest_hash, "manifest_hash")
        timestamp = _now(now)
        with self._write(connection) as session:
            job = session.scalar(select(Job).where(Job.job_id == job_id).with_for_update())
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if job.seal_hash is not None:
                if job.seal_hash != manifest_hash:
                    raise conflict("job was sealed with a different manifest")
                return _decode(job), True
            if job.state != JobState.UPLOADING.value:
                raise failed_precondition("job cannot be sealed from its current state")
            active = session.scalar(
                select(InputUpload.upload_token).where(InputUpload.job_id == job_id).limit(1)
            )
            if active is not None:
                raise failed_precondition("job has an upload in progress")
            job.state = JobState.SEALED.value
            job.revision += 1
            job.updated_at = timestamp
            job.sealed_at = timestamp
            job.seal_hash = manifest_hash
            job.seal_manifest = _json_value(manifest)
            job.seal_result = _json_value(result)
            job.source_width = source_width
            job.feature_dim = feature_dim
            session.flush()
            return _decode(job), False

    def queue_job(
        self,
        job_id: str,
        *,
        selected_device: str,
        result: dict | None = None,
        now: float | None = None,
        connection: Session | None = None,
    ) -> tuple[dict, bool]:
        if selected_device not in ("cpu", "cuda"):
            raise ValueError("selected_device must be cpu or cuda")
        timestamp = _now(now)
        with self._write(connection) as session:
            job = session.scalar(select(Job).where(Job.job_id == job_id).with_for_update())
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if job.queued_at is not None:
                if job.selected_device != selected_device:
                    raise conflict("job was started with a different device")
                return _decode(job), True
            if job.state != JobState.SEALED.value:
                raise failed_precondition("job must be SEALED before start")
            job.state = JobState.QUEUED.value
            job.revision += 1
            job.selected_device = selected_device
            job.start_result = _json_value(result)
            job.queued_at = timestamp
            job.updated_at = timestamp
            job.queue_sequence = session.scalar(select(QUEUE_SEQUENCE.next_value()))
            session.flush()
            return _decode(job), False

    def claim_job(
        self,
        job_id: str,
        selected_device: str,
        *,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> dict | None:
        if selected_device not in ("cpu", "cuda"):
            raise ValueError("selected_device must be cpu or cuda")
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job).where(Job.job_id == job_id).with_for_update(skip_locked=True)
            )
            if job is None or job.state != JobState.QUEUED.value or job.selected_device != selected_device:
                return None
            return self._claim(session, job, worker_id, _now(now))

    def claim_next_job(
        self,
        selected_device: str,
        *,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> dict | None:
        if selected_device not in ("cpu", "cuda"):
            raise ValueError("selected_device must be cpu or cuda")
        with self.database.transaction() as session:
            job = session.scalar(
                select(Job)
                .where(Job.state == JobState.QUEUED.value, Job.selected_device == selected_device)
                .order_by(Job.queue_sequence)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                return None
            return self._claim(session, job, worker_id, _now(now))

    def _claim(self, session: Session, job: Job, worker_id: str | None, timestamp: datetime) -> dict:
        job.state = JobState.RUNNING.value
        job.revision += 1
        job.attempt += 1
        job.started_at = timestamp
        job.updated_at = timestamp
        session.add(JobAttempt(
            job_id=job.job_id,
            attempt=job.attempt,
            selected_device=job.selected_device,
            status=JobState.RUNNING.value,
            worker_id=worker_id,
            claimed_at=timestamp,
            started_at=timestamp,
        ))
        session.flush()
        return _decode(job)

    def set_attempt_process(
        self,
        job_id: str,
        attempt: int,
        *,
        pid: int,
        pgid: int,
        boot_id: str,
        process_start_ticks: int,
    ) -> None:
        _positive(pid, "pid")
        _positive(pgid, "pgid")
        _positive(process_start_ticks, "process_start_ticks")
        boot_id = _canonical_uuid(boot_id, "boot_id")
        with self.database.transaction() as session:
            record = session.get(JobAttempt, (job_id, attempt), with_for_update=True)
            if record is None or record.status != JobState.RUNNING.value:
                raise failed_precondition("job attempt is not running")
            record.pid = pid
            record.pgid = pgid
            record.boot_id = boot_id
            record.process_start_ticks = process_start_ticks

    def list_active_attempts(self) -> list[dict]:
        with self.database.session() as session:
            rows = session.scalars(
                select(JobAttempt)
                .join(Job, Job.job_id == JobAttempt.job_id)
                .where(
                    Job.state.in_((JobState.RUNNING.value, JobState.CANCELLING.value)),
                    JobAttempt.status == JobState.RUNNING.value,
                    JobAttempt.attempt == Job.attempt,
                )
                .order_by(JobAttempt.job_id, JobAttempt.attempt)
            )
            return [_decode(row) for row in rows]

    def finish_attempt(
        self,
        job_id: str,
        attempt: int,
        target_state: str | JobState,
        *,
        error_code: str | ErrorCode | None = None,
        error_message: str | None = None,
        exit_code: int | None = None,
        now: float | None = None,
    ) -> dict:
        target_state = JobState(target_state)
        if target_state not in (JobState.FAILED, JobState.CANCELLED):
            raise ValueError("worker attempt can finish only as FAILED or CANCELLED")
        if target_state == JobState.CANCELLED and (error_code is not None or error_message is not None):
            raise ValueError("cancelled attempt must not carry an error")
        timestamp = _now(now)
        with self.database.transaction() as session:
            job = session.scalar(select(Job).where(Job.job_id == job_id).with_for_update())
            if job is None:
                raise not_found(f"job not found: {job_id}")
            if job.attempt != attempt:
                raise failed_precondition("job attempt is no longer active")
            validate_transition(job.state, target_state)
            record = session.get(JobAttempt, (job_id, attempt), with_for_update=True)
            if record is None or record.status != JobState.RUNNING.value:
                raise failed_precondition("job attempt is no longer active")
            code = error_code.value if isinstance(error_code, ErrorCode) else error_code
            record.status = target_state.value
            record.finished_at = timestamp
            record.exit_code = exit_code
            record.error_code = code
            record.error_message = error_message
            job.state = target_state.value
            job.revision += 1
            job.error_code = code
            job.error_message = error_message
            job.finished_at = timestamp
            job.updated_at = timestamp
            session.flush()
            return _decode(job)

    def publish_outputs(
        self,
        job_id: str,
        attempt: int,
        outputs: Sequence[dict],
        *,
        result: dict,
        now: float | None = None,
    ) -> dict:
        timestamp = _now(now)
        try:
            with self.database.transaction() as session:
                job = session.scalar(select(Job).where(Job.job_id == job_id).with_for_update())
                if job is None:
                    raise not_found(f"job not found: {job_id}")
                if job.operation != "predict" or job.state != JobState.RUNNING.value or job.attempt != attempt:
                    raise failed_precondition("job is not the active running attempt")
                for output in outputs:
                    session.add(_output_record(job_id, output, timestamp))
                session.flush()
                record = session.get(JobAttempt, (job_id, attempt), with_for_update=True)
                record.status = JobState.SUCCEEDED.value
                record.finished_at = timestamp
                job.state = JobState.SUCCEEDED.value
                job.revision += 1
                job.result = _json_value(result)
                job.error_code = None
                job.error_message = None
                job.finished_at = timestamp
                job.updated_at = timestamp
                session.flush()
                return _decode(job)
        except IntegrityError as exc:
            raise conflict("job output ordinal is already published") from exc

    def publish_model(
        self,
        job_id: str,
        attempt: int,
        *,
        model_ref: str,
        label: str,
        generation: int | None,
        checkpoint_path: str,
        metadata_path: str,
        sha256: str,
        metadata: dict,
        result: dict,
        now: float | None = None,
    ) -> dict:
        _validate_relative_path(checkpoint_path)
        _validate_relative_path(metadata_path)
        _digest(sha256, "sha256")
        timestamp = _now(now)
        try:
            with self.database.transaction() as session:
                job = session.scalar(select(Job).where(Job.job_id == job_id).with_for_update())
                if job is None:
                    raise not_found(f"job not found: {job_id}")
                if job.operation != "fit" or job.state != JobState.RUNNING.value or job.attempt != attempt:
                    raise failed_precondition("job is not the active fit attempt")
                if label != job.model_label:
                    raise failed_precondition("model label does not match the fit job")
                _advisory_lock(session, "model-generation", job.owner_subject, label)
                next_generation = int(session.scalar(
                    select(func.coalesce(func.max(PublishedModel.generation), 0) + 1).where(
                        PublishedModel.owner_subject == job.owner_subject,
                        PublishedModel.label == label,
                    )
                ))
                if generation is None:
                    generation = next_generation
                elif generation != next_generation:
                    raise failed_precondition(
                        f"next model generation is {next_generation}, got {generation}"
                    )
                session.add(PublishedModel(
                    model_ref=model_ref,
                    owner_subject=job.owner_subject,
                    label=label,
                    generation=generation,
                    checkpoint_path=checkpoint_path,
                    metadata_path=metadata_path,
                    sha256=sha256,
                    metadata_json=_json_value(metadata),
                    producing_job_id=job_id,
                    created_at=timestamp,
                ))
                alias = session.get(ModelAlias, (job.owner_subject, label), with_for_update=True)
                if alias is None:
                    session.add(ModelAlias(
                        owner_subject=job.owner_subject,
                        label=label,
                        model_ref=model_ref,
                        updated_at=timestamp,
                    ))
                else:
                    alias.model_ref = model_ref
                    alias.updated_at = timestamp
                record = session.get(JobAttempt, (job_id, attempt), with_for_update=True)
                record.status = JobState.SUCCEEDED.value
                record.finished_at = timestamp
                job.state = JobState.SUCCEEDED.value
                job.revision += 1
                job.result = _json_value(result)
                job.error_code = None
                job.error_message = None
                job.finished_at = timestamp
                job.updated_at = timestamp
                session.flush()
                return _decode(job)
        except IntegrityError as exc:
            raise conflict("model generation already exists") from exc

    def get_model(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
    ) -> dict | None:
        statement = select(PublishedModel).where(PublishedModel.model_ref == model_ref)
        if owner_subject is not None:
            statement = statement.where(PublishedModel.owner_subject == owner_subject)
        with self._read(connection) as session:
            return _decode(session.scalar(statement))

    def resolve_model_alias(
        self,
        owner_subject: str,
        label: str,
        *,
        connection: Session | None = None,
    ) -> dict | None:
        with self._read(connection) as session:
            model = session.scalar(
                select(PublishedModel)
                .join(ModelAlias, ModelAlias.model_ref == PublishedModel.model_ref)
                .where(ModelAlias.owner_subject == owner_subject, ModelAlias.label == label)
            )
            return _decode(model)

    def list_models(self) -> list[dict]:
        with self.database.session() as session:
            rows = session.scalars(
                select(PublishedModel).order_by(
                    PublishedModel.owner_subject,
                    PublishedModel.label,
                    PublishedModel.generation,
                )
            )
            return [_decode(row) for row in rows]

    def list_outputs(self, job_id: str, *, connection: Session | None = None) -> list[dict]:
        with self._read(connection) as session:
            rows = session.scalars(
                select(JobOutput).where(JobOutput.job_id == job_id).order_by(JobOutput.ordinal)
            )
            return [_decode(row) for row in rows]

    def issue_ticket(
        self,
        *,
        job_id: str,
        ordinal: int,
        owner_subject: str,
        ttl_seconds: float,
        now: float | None = None,
    ) -> tuple[bytes, float]:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        timestamp = _now(now)
        expires_at = datetime.fromtimestamp(timestamp.timestamp() + ttl_seconds, timezone.utc)
        token = secrets.token_urlsafe(32).encode("ascii")
        ticket_hash = hashlib.sha256(token).hexdigest()
        with self.database.transaction() as session:
            output = session.scalar(
                select(JobOutput)
                .join(Job, Job.job_id == JobOutput.job_id)
                .where(
                    JobOutput.job_id == job_id,
                    JobOutput.ordinal == ordinal,
                    Job.owner_subject == owner_subject,
                    Job.state == JobState.SUCCEEDED.value,
                )
            )
            if output is None:
                raise not_found("published job output not found")
            session.add(OutputTicket(
                ticket_hash=ticket_hash,
                job_id=job_id,
                ordinal=ordinal,
                owner_subject=owner_subject,
                expires_at=expires_at,
                created_at=timestamp,
            ))
        return token, expires_at.timestamp()

    def resolve_ticket(
        self,
        ticket: bytes,
        *,
        owner_subject: str,
        now: float | None = None,
    ) -> dict:
        ticket_hash = hashlib.sha256(bytes(ticket)).hexdigest()
        with self.database.session() as session:
            row = session.execute(
                select(OutputTicket, JobOutput, Job.state)
                .join(
                    JobOutput,
                    and_(
                        JobOutput.job_id == OutputTicket.job_id,
                        JobOutput.ordinal == OutputTicket.ordinal,
                    ),
                )
                .join(Job, Job.job_id == OutputTicket.job_id)
                .where(OutputTicket.ticket_hash == ticket_hash)
            ).one_or_none()
        if row is None:
            raise not_found("output ticket not found")
        record, output, state = row
        if record.owner_subject != owner_subject:
            raise ServiceError(ErrorCode.PERMISSION_DENIED, "output ticket belongs to another subject")
        if record.expires_at <= _now(now):
            raise failed_precondition("output ticket has expired")
        if state != JobState.SUCCEEDED.value:
            raise failed_precondition("job output is not available")
        result = _decode(output)
        result.update({
            "ticket_owner": record.owner_subject,
            "expires_at": record.expires_at.timestamp(),
            "state": state,
        })
        return result

    def delete_expired_tickets(self, *, now: float | None = None) -> int:
        with self.database.transaction() as session:
            result = session.execute(delete(OutputTicket).where(OutputTicket.expires_at <= _now(now)))
            return result.rowcount

    def reconcile_interrupted_jobs(self, *, now: float | None = None) -> dict:
        timestamp = _now(now)
        with self.database.transaction() as session:
            uploads = list(session.scalars(select(InputUpload.temporary_path)))
            interrupted = list(session.scalars(
                select(Job.job_id).where(Job.state == JobState.RUNNING.value).order_by(Job.job_id)
            ))
            cancelling = list(session.scalars(
                select(Job.job_id).where(Job.state == JobState.CANCELLING.value).order_by(Job.job_id)
            ))
            for job in session.scalars(
                select(Job).where(Job.state.in_((JobState.RUNNING.value, JobState.CANCELLING.value))).with_for_update()
            ):
                attempt = session.get(JobAttempt, (job.job_id, job.attempt), with_for_update=True)
                if job.state == JobState.RUNNING.value:
                    job.state = JobState.FAILED.value
                    job.error_code = ErrorCode.EXECUTION_INTERRUPTED.value
                    job.error_message = "worker execution was interrupted by service restart"
                    if attempt is not None and attempt.status == JobState.RUNNING.value:
                        attempt.status = JobState.FAILED.value
                        attempt.error_code = job.error_code
                        attempt.error_message = job.error_message
                        attempt.finished_at = timestamp
                else:
                    job.state = JobState.CANCELLED.value
                    if attempt is not None and attempt.status == JobState.RUNNING.value:
                        attempt.status = JobState.CANCELLED.value
                        attempt.finished_at = timestamp
                job.revision += 1
                job.finished_at = timestamp
                job.updated_at = timestamp
            session.execute(delete(InputUpload))
        return {
            "interrupted_jobs": interrupted,
            "cancelled_jobs": cancelling,
            "temporary_paths": uploads,
        }

    def referenced_paths(self) -> set[str]:
        with self.database.session() as session:
            paths = set(session.scalars(select(JobInput.relative_path)))
            paths.update(session.scalars(select(JobOutput.relative_path)))
            return paths

    def delete_terminal_jobs_before(self, cutoff: float) -> list[str]:
        cutoff_at = _at(cutoff)
        with self.database.transaction() as session:
            candidates = session.scalars(
                select(Job)
                .where(
                    Job.state.in_(_TERMINAL_STATES),
                    Job.finished_at.is_not(None),
                    Job.finished_at < cutoff_at,
                    ~select(PublishedModel.model_ref)
                    .where(PublishedModel.producing_job_id == Job.job_id)
                    .exists(),
                    ~select(OutputTicket.ticket_hash)
                    .where(OutputTicket.job_id == Job.job_id)
                    .exists(),
                    ~select(IdempotencyRecord.idempotency_key)
                    .where(
                        IdempotencyRecord.job_id == Job.job_id,
                        IdempotencyRecord.created_at >= cutoff_at,
                    )
                    .exists(),
                )
                .order_by(Job.job_id)
                .with_for_update()
            ).all()
            job_ids = [job.job_id for job in candidates]
            if job_ids:
                session.execute(delete(IdempotencyRecord).where(IdempotencyRecord.job_id.in_(job_ids)))
                session.execute(delete(Job).where(Job.job_id.in_(job_ids)))
            session.execute(delete(IdempotencyRecord).where(
                IdempotencyRecord.job_id.is_(None),
                IdempotencyRecord.created_at < cutoff_at,
            ))
            return job_ids

    def synchronize_runtime_epoch(self, epoch: str, *, now: float | None = None) -> dict:
        epoch = _canonical_uuid(epoch, "runtime storage epoch")
        timestamp = _now(now)
        with self.database.transaction() as session:
            state = session.get(RuntimeState, "storage_epoch", with_for_update=True)
            if state is not None and state.value == epoch:
                return {"reset": False, "discarded_jobs": []}
            discarded_jobs = list(session.scalars(select(Job.job_id).order_by(Job.job_id)))
            reset = state is not None or bool(discarded_jobs)
            session.execute(delete(OutputTicket))
            session.execute(delete(IdempotencyRecord))
            session.execute(delete(InputUpload))
            session.execute(delete(Job))
            if state is None:
                session.add(RuntimeState(key="storage_epoch", value=epoch, updated_at=timestamp))
            else:
                state.value = epoch
                state.updated_at = timestamp
            return {"reset": reset, "discarded_jobs": discarded_jobs}

    def queued_jobs(self) -> list[dict]:
        with self.database.session() as session:
            rows = session.scalars(
                select(Job)
                .where(Job.state == JobState.QUEUED.value)
                .order_by(Job.queue_sequence)
            )
            return [_decode(row) for row in rows]


def _output_record(job_id: str, output: dict, timestamp: datetime) -> JobOutput:
    relative_path = output["relative_path"]
    _validate_relative_path(relative_path)
    for name in ("ordinal", "rows", "batches", "bytes"):
        _nonnegative(output[name], name)
    _digest(output["sha256"], "sha256")
    _digest(output["schema_fingerprint"], "schema_fingerprint")
    return JobOutput(
        job_id=job_id,
        ordinal=output["ordinal"],
        rows=output["rows"],
        batches=output["batches"],
        bytes=output["bytes"],
        sha256=output["sha256"],
        schema_fingerprint=output["schema_fingerprint"],
        relative_path=relative_path,
        published_at=timestamp,
    )


def _transition_updates(updates: dict) -> dict:
    mapping = {
        "selected_device": "selected_device",
        "sealed_at": "sealed_at",
        "queued_at": "queued_at",
        "started_at": "started_at",
        "cancel_requested_at": "cancel_requested_at",
        "finished_at": "finished_at",
        "error_code": "error_code",
        "error_message": "error_message",
        "result_json": "result",
        "progress_json": "progress",
    }
    unknown = set(updates) - set(mapping)
    if unknown:
        raise ValueError(f"unsupported job update field(s): {', '.join(sorted(unknown))}")
    encoded = {}
    for key, value in updates.items():
        target = mapping[key]
        if key in ("result_json", "progress_json") and value is not None:
            value = _json_value(value)
        elif key.endswith("_at") and value is not None and not isinstance(value, datetime):
            value = _at(value)
        encoded[target] = value
    return encoded


def _decode(record) -> dict | None:
    if record is None:
        return None
    result = {}
    for attribute in record.__mapper__.column_attrs:
        column = attribute.columns[0]
        value = getattr(record, attribute.key)
        if isinstance(value, datetime):
            value = value.timestamp()
        result[column.name] = value
    return result


def _json_value(value):
    if value is None:
        return None
    if is_dataclass(value):
        value = asdict(value)
    return json.loads(json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ))


def _canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value.lower():
        raise ValueError(f"{label} must be a canonical UUID")
    return str(parsed)


def _validate_relative_path(value: str) -> None:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        raise ValueError("artifact path must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("artifact path must not contain path traversal")


def _nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _now(value: float | None) -> datetime:
    return datetime.now(timezone.utc) if value is None else _at(value)


def _at(value: float) -> datetime:
    return datetime.fromtimestamp(float(value), timezone.utc)


def _advisory_lock(session: Session, *parts: str) -> None:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    key = int.from_bytes(digest[:8], "big", signed=True)
    session.scalar(select(func.pg_advisory_xact_lock(key)))
