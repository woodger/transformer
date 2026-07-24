from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.migrations import require_current_schema
from app.database.models import (
    IdempotencyRecord,
    Job,
    JobAttempt,
    JobInput,
    JobOutput,
    TrainingRecoveryCheckpoint,
)
from app.database.session import Database
from app.flight.constants import (
    SUPPORTED_DEVICES,
    SUPPORTED_OPERATIONS,
    ErrorCode,
    JobState,
)
from app.flight.errors import conflict, failed_precondition, not_found
from app.flight.ledger_artifacts import ArtifactLedgerSlice
from app.flight.ledger_execution import ExecutionLedgerSlice
from app.flight.ledger_inputs import InputLedgerSlice
from app.flight.ledger_maintenance import MaintenanceLedgerSlice
from app.flight.ledger_recovery import RecoveryLedgerSlice
from app.flight.ledger_support import (
    LedgerSessions,
    advisory_lock as _advisory_lock,
    at as _at,
    canonical_uuid as _canonical_uuid,
    decode as _decode,
    digest as _digest,
    json_value as _json_value,
    now as _now,
)
from app.flight.records import (
    CommittedInputRecord,
    ExecutionJobRecord,
    ModelArtifactRecord,
    RecoverableAttemptRecord,
    StatusRecoveryRecord,
    TrainingRecoveryCheckpointRecord,
)
from app.flight.state import validate_transition

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
        self._sessions = LedgerSessions(database)
        self._inputs = InputLedgerSlice(self._sessions)
        self._execution = ExecutionLedgerSlice(self._sessions)
        self._artifacts = ArtifactLedgerSlice(self._sessions)
        self._maintenance = MaintenanceLedgerSlice(self._sessions)
        self._recovery = RecoveryLedgerSlice(self._sessions)

    def initialize(self) -> Ledger:
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
        with self._sessions.read(connection) as session:
            yield session

    @contextmanager
    def _write(self, connection: Session | None):
        with self._sessions.write(connection) as session:
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

    def get_execution_job(
        self,
        job_id: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
        for_update: bool = False,
    ) -> ExecutionJobRecord | None:
        return self._execution.get_execution_job(
            job_id,
            owner_subject=owner_subject,
            connection=connection,
            for_update=for_update,
        )

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
            recovery = None
            if job is not None and job.operation == "fit":
                checkpoint = session.scalar(
                    select(TrainingRecoveryCheckpoint)
                    .where(
                        TrainingRecoveryCheckpoint.job_id
                        == job_id
                    )
                    .order_by(
                        TrainingRecoveryCheckpoint.generation.desc()
                    )
                    .limit(1)
                )
                last_retry = session.scalar(
                    select(JobAttempt)
                    .where(
                        JobAttempt.job_id == job_id,
                        JobAttempt.error_code.in_((
                            ErrorCode.DEVICE_LOST.value,
                            ErrorCode.EXECUTION_INTERRUPTED.value,
                        )),
                    )
                    .order_by(JobAttempt.attempt.desc())
                    .limit(1)
                )
                retry_count = int(session.scalar(
                    select(func.count())
                    .select_from(JobAttempt)
                    .where(
                        JobAttempt.job_id == job_id,
                        JobAttempt.error_code.in_((
                            ErrorCode.DEVICE_LOST.value,
                            ErrorCode.EXECUTION_INTERRUPTED.value,
                        )),
                    )
                ) or 0)
                active_attempt = (
                    None
                    if job.attempt <= 0
                    else session.get(
                        JobAttempt,
                        (job_id, job.attempt),
                    )
                )
                recovery = StatusRecoveryRecord(
                    checkpoint=(
                        None
                        if checkpoint is None
                        else TrainingRecoveryCheckpointRecord(
                            job_id=checkpoint.job_id,
                            generation=checkpoint.generation,
                            attempt=checkpoint.attempt,
                            format=checkpoint.format,
                            relative_path=checkpoint.relative_path,
                            byte_count=checkpoint.bytes,
                            sha256=checkpoint.sha256,
                            completed_epochs=checkpoint.completed_epochs,
                            global_step=checkpoint.global_step,
                            training_complete=checkpoint.training_complete,
                        )
                    ),
                    retry_count=retry_count,
                    last_retry_code=(
                        None
                        if last_retry is None
                        else last_retry.error_code
                    ),
                    resumed_from_generation=(
                        None
                        if active_attempt is None
                        else active_attempt.resume_generation
                    ),
                )
        return (
            _decode(job),
            [_decode(row) for row in inputs],
            [_decode(row) for row in outputs],
            recovery,
        )

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
        return self._execution.update_progress(
            job_id,
            progress,
            now=now,
        )

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
        return self._inputs.find_input(
            job_id,
            ordinal=ordinal,
            payload_id=payload_id,
            connection=connection,
        )

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
        return self._inputs.reserve_input(
            job_id=job_id,
            payload_id=payload_id,
            ordinal=ordinal,
            upload_token=upload_token,
            temporary_path=temporary_path,
            storage_class=storage_class,
            now=now,
        )

    def abort_input(self, upload_token: str) -> str | None:
        return self._inputs.abort_input(upload_token)

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
        return self._inputs.commit_input(
            upload_token=upload_token,
            relative_path=relative_path,
            schema_id=schema_id,
            rows=rows,
            batches=batches,
            byte_count=byte_count,
            sha256=sha256,
            schema_fingerprint=schema_fingerprint,
            source_width=source_width,
            feature_dim=feature_dim,
            max_payloads=max_payloads,
            max_job_bytes=max_job_bytes,
            storage_class=storage_class,
            now=now,
        )

    def list_inputs(
        self,
        job_id: str,
        *,
        connection: Session | None = None,
    ) -> list[dict]:
        return self._inputs.list_inputs(job_id, connection=connection)

    def list_committed_inputs(
        self,
        job_id: str,
        *,
        connection: Session | None = None,
    ) -> list[CommittedInputRecord]:
        return self._inputs.list_committed_inputs(
            job_id,
            connection=connection,
        )

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
        return self._inputs.seal_job(
            job_id,
            manifest_hash=manifest_hash,
            manifest=manifest,
            source_width=source_width,
            feature_dim=feature_dim,
            result=result,
            now=now,
            connection=connection,
        )

    def queue_job(
        self,
        job_id: str,
        *,
        selected_device: str,
        result: dict | None = None,
        now: float | None = None,
        connection: Session | None = None,
    ) -> tuple[dict, bool]:
        return self._execution.queue_job(
            job_id,
            selected_device=selected_device,
            result=result,
            now=now,
            connection=connection,
        )

    def claim_job(
        self,
        job_id: str,
        selected_device: str,
        *,
        worker_id: str | None = None,
        device_id: str | None = None,
        now: float | None = None,
    ) -> dict | None:
        return self._execution.claim_job(
            job_id,
            selected_device,
            worker_id=worker_id,
            device_id=device_id,
            now=now,
        )

    def claim_execution_job(
        self,
        job_id: str,
        selected_device: str,
        *,
        worker_id: str | None = None,
        device_id: str | None = None,
        now: float | None = None,
    ) -> ExecutionJobRecord | None:
        return self._execution.claim_execution_job(
            job_id,
            selected_device,
            worker_id=worker_id,
            device_id=device_id,
            now=now,
        )

    def claim_next_job(
        self,
        selected_device: str,
        *,
        worker_id: str | None = None,
        device_id: str | None = None,
        now: float | None = None,
    ) -> dict | None:
        return self._execution.claim_next_job(
            selected_device,
            worker_id=worker_id,
            device_id=device_id,
            now=now,
        )

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
        self._execution.set_attempt_process(
            job_id,
            attempt,
            pid=pid,
            pgid=pgid,
            boot_id=boot_id,
            process_start_ticks=process_start_ticks,
        )

    def list_active_attempts(self) -> list[dict]:
        return self._execution.list_active_attempts()

    def list_recoverable_attempts(self) -> list[RecoverableAttemptRecord]:
        return self._execution.list_recoverable_attempts()

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
        return self._execution.finish_attempt(
            job_id,
            attempt,
            target_state,
            error_code=error_code,
            error_message=error_message,
            exit_code=exit_code,
            now=now,
        )

    def register_recovery_checkpoint(
        self,
        *,
        job_id: str,
        attempt: int,
        generation: int,
        format: str,
        relative_path: str,
        byte_count: int,
        sha256: str,
        completed_epochs: int,
        global_step: int,
        training_complete: bool,
        now: float | None = None,
    ) -> tuple[TrainingRecoveryCheckpointRecord, bool]:
        return self._recovery.register_checkpoint(
            job_id=job_id,
            attempt=attempt,
            generation=generation,
            format=format,
            relative_path=relative_path,
            byte_count=byte_count,
            sha256=sha256,
            completed_epochs=completed_epochs,
            global_step=global_step,
            training_complete=training_complete,
            now=now,
        )

    def latest_recovery_checkpoint(
        self,
        job_id: str,
    ) -> TrainingRecoveryCheckpointRecord | None:
        return self._recovery.latest_checkpoint(job_id)

    def list_recovery_checkpoints(
        self,
        job_id: str,
    ) -> list[TrainingRecoveryCheckpointRecord]:
        return self._recovery.list_checkpoints(job_id)

    def recovery_referenced_paths(self) -> set[str]:
        return self._recovery.referenced_paths()

    def active_recovery_job_ids(self) -> set[str]:
        return self._recovery.active_fit_job_ids()

    def terminal_recovery_job_ids(self) -> set[str]:
        return self._recovery.terminal_fit_job_ids()

    def prune_recovery_checkpoints(
        self,
        job_id: str,
        *,
        keep: int = 2,
    ) -> list[str]:
        return self._recovery.prune_checkpoints(
            job_id,
            keep=keep,
        )

    def schedule_retry(
        self,
        job_id: str,
        attempt: int,
        *,
        error_code: str | ErrorCode,
        error_message: str,
        exit_code: int | None = None,
        now: float | None = None,
    ) -> dict:
        return self._recovery.schedule_retry(
            job_id,
            attempt,
            error_code=error_code,
            error_message=error_message,
            exit_code=exit_code,
            now=now,
        )

    def publish_outputs(
        self,
        job_id: str,
        attempt: int,
        outputs: Sequence[dict],
        *,
        result: dict,
        now: float | None = None,
    ) -> dict:
        return self._artifacts.publish_outputs(
            job_id,
            attempt,
            outputs,
            result=result,
            now=now,
        )

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
        return self._artifacts.publish_model(
            job_id,
            attempt,
            model_ref=model_ref,
            label=label,
            generation=generation,
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            sha256=sha256,
            metadata=metadata,
            result=result,
            now=now,
        )

    def get_model(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
    ) -> dict | None:
        return self._artifacts.get_model(
            model_ref,
            owner_subject=owner_subject,
            connection=connection,
        )

    def get_model_artifact(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
    ) -> ModelArtifactRecord | None:
        return self._artifacts.get_model_artifact(
            model_ref,
            owner_subject=owner_subject,
            connection=connection,
        )

    def resolve_model_alias(
        self,
        owner_subject: str,
        label: str,
        *,
        connection: Session | None = None,
    ) -> dict | None:
        return self._artifacts.resolve_model_alias(
            owner_subject,
            label,
            connection=connection,
        )

    def list_models(self) -> list[dict]:
        return self._artifacts.list_models()

    def list_outputs(self, job_id: str, *, connection: Session | None = None) -> list[dict]:
        return self._artifacts.list_outputs(
            job_id,
            connection=connection,
        )

    def issue_ticket(
        self,
        *,
        job_id: str,
        ordinal: int,
        owner_subject: str,
        ttl_seconds: float,
        now: float | None = None,
    ) -> tuple[bytes, float]:
        return self._artifacts.issue_ticket(
            job_id=job_id,
            ordinal=ordinal,
            owner_subject=owner_subject,
            ttl_seconds=ttl_seconds,
            now=now,
        )

    def resolve_ticket(
        self,
        ticket: bytes,
        *,
        owner_subject: str,
        now: float | None = None,
    ) -> dict:
        return self._artifacts.resolve_ticket(
            ticket,
            owner_subject=owner_subject,
            now=now,
        )

    def delete_expired_tickets(self, *, now: float | None = None) -> int:
        return self._maintenance.delete_expired_tickets(now=now)

    def reconcile_interrupted_jobs(self, *, now: float | None = None) -> dict:
        return self._maintenance.reconcile_interrupted_jobs(now=now)

    def referenced_paths(self) -> set[str]:
        return self._maintenance.referenced_paths()

    def delete_terminal_jobs_before(self, cutoff: float) -> list[str]:
        return self._maintenance.delete_terminal_jobs_before(cutoff)

    def synchronize_runtime_epoch(self, epoch: str, *, now: float | None = None) -> dict:
        return self._maintenance.synchronize_runtime_epoch(
            epoch,
            now=now,
        )

    def queued_jobs(self) -> list[dict]:
        return self._execution.queued_jobs()

    def queued_execution_jobs(self) -> list[ExecutionJobRecord]:
        return self._execution.queued_execution_jobs()


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
        raise ValueError(
            f"unsupported job update field(s): {', '.join(sorted(unknown))}"
        )
    encoded = {}
    for key, value in updates.items():
        target = mapping[key]
        if key in ("result_json", "progress_json") and value is not None:
            value = _json_value(value)
        elif (
            key.endswith("_at")
            and value is not None
            and not isinstance(value, datetime)
        ):
            value = _at(value)
        encoded[target] = value
    return encoded
