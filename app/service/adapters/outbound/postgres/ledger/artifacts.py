from __future__ import annotations

import hashlib
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.json_types import JsonObject
from app.service.adapters.outbound.postgres.ledger.support import (
    LedgerSessions,
    RowMapping,
    advisory_lock,
    canonical_uuid,
    decode,
    decode_optional,
    digest,
    json_value,
    nonnegative,
    now as timestamp_now,
    validate_relative_path,
)
from app.service.adapters.outbound.postgres.mapping import published_model_record
from app.service.adapters.outbound.postgres.models import (
    DeletedModel,
    Job,
    JobAttempt,
    JobOutput,
    ModelAlias,
    OutputTicket,
    PublishedModel,
)
from app.service.domain.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    not_found,
)
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.model import ModelLifecycleState
from app.service.domain.records import ModelArtifactRecord, PublishedModelRecord


class ArtifactLedgerSlice:
    """Atomic publication and retrieval of Transformer-owned artifacts."""

    def __init__(self, sessions: LedgerSessions) -> None:
        self.sessions = sessions
        self.database = sessions.database

    def publish_outputs(
        self,
        job_id: str,
        attempt: int,
        outputs: Sequence[JsonObject],
        *,
        attempt_id: str,
        result: JsonObject,
        now: float | None = None,
    ) -> RowMapping:
        published_at = timestamp_now(now)
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        try:
            with self.database.transaction() as session:
                job = session.scalar(
                    select(Job)
                    .where(Job.job_id == job_id)
                    .with_for_update()
                )
                if job is None:
                    raise not_found(f"job not found: {job_id}")
                if (
                    job.operation != "predict"
                    or job.execution_state != ExecutionState.RUNNING.value
                    or job.input_state != InputState.CLOSED.value
                    or job.attempt != attempt
                ):
                    raise failed_precondition(
                        "job is not the active running attempt"
                    )
                record = session.get(
                    JobAttempt,
                    (job_id, attempt),
                    with_for_update=True,
                )
                if (
                    record is None
                    or record.status != ExecutionState.RUNNING.value
                    or record.attempt_id != attempt_id
                ):
                    raise failed_precondition(
                        "job attempt is no longer active"
                    )
                for output in outputs:
                    session.add(
                        _output_record(job_id, output, published_at)
                    )
                session.flush()
                record.status = ExecutionState.SUCCEEDED.value
                record.finished_at = published_at
                job.execution_state = ExecutionState.SUCCEEDED.value
                job.revision += 1
                job.result = json_value(result)
                job.error_code = None
                job.error_message = None
                job.finished_at = published_at
                job.updated_at = published_at
                session.flush()
                return decode(job)
        except IntegrityError as exc:
            raise conflict(
                "job output ordinal is already published"
            ) from exc

    def publish_model(
        self,
        job_id: str,
        attempt: int,
        *,
        attempt_id: str,
        model_ref: str,
        label: str,
        generation: int | None,
        checkpoint_path: str,
        metadata_path: str,
        byte_count: int,
        sha256: str,
        metadata: JsonObject,
        result: JsonObject,
        now: float | None = None,
    ) -> RowMapping:
        attempt_id = canonical_uuid(attempt_id, "attempt_id")
        validate_relative_path(checkpoint_path)
        validate_relative_path(metadata_path)
        if isinstance(byte_count, bool) or byte_count <= 0:
            raise ValueError("byte_count must be a positive integer")
        digest(sha256, "sha256")
        published_at = timestamp_now(now)
        try:
            with self.database.transaction() as session:
                job = session.scalar(
                    select(Job)
                    .where(Job.job_id == job_id)
                    .with_for_update()
                )
                if job is None:
                    raise not_found(f"job not found: {job_id}")
                if (
                    job.operation != "fit"
                    or job.execution_state != ExecutionState.RUNNING.value
                    or job.input_state != InputState.CLOSED.value
                    or job.attempt != attempt
                ):
                    raise failed_precondition(
                        "job is not the active fit attempt"
                    )
                record = session.get(
                    JobAttempt,
                    (job_id, attempt),
                    with_for_update=True,
                )
                if (
                    record is None
                    or record.status != ExecutionState.RUNNING.value
                    or record.attempt_id != attempt_id
                ):
                    raise failed_precondition(
                        "job attempt is no longer active"
                    )
                if label != job.model_label:
                    raise failed_precondition(
                        "model label does not match the fit job"
                    )
                advisory_lock(
                    session,
                    "model-generation",
                    job.owner_subject,
                    label,
                )
                if session.get(DeletedModel, model_ref) is not None:
                    raise conflict(
                        "model reference belongs to a deleted generation"
                    )
                current_generation = session.scalar(
                    select(func.max(PublishedModel.generation)).where(
                        PublishedModel.owner_subject == job.owner_subject,
                        PublishedModel.label == label,
                    )
                )
                deleted_generation = session.scalar(
                    select(func.max(DeletedModel.generation)).where(
                        DeletedModel.owner_subject == job.owner_subject,
                        DeletedModel.label == label,
                    )
                )
                next_generation = max(
                    0 if current_generation is None else current_generation,
                    0 if deleted_generation is None else deleted_generation,
                ) + 1
                if generation is None:
                    generation = next_generation
                elif generation != next_generation:
                    raise failed_precondition(
                        f"next model generation is {next_generation}, "
                        f"got {generation}"
                    )
                session.add(PublishedModel(
                    model_ref=model_ref,
                    owner_subject=job.owner_subject,
                    label=label,
                    generation=generation,
                    checkpoint_path=checkpoint_path,
                    metadata_path=metadata_path,
                    checkpoint_bytes=byte_count,
                    sha256=sha256,
                    metadata_json=json_value(metadata),
                    data_contract=json_value(job.data_contract),
                    data_contract_sha256=job.data_contract_sha256,
                    ml_contract=json_value(job.ml_contract),
                    objective_config_sha256=digest(
                        job.ml_contract.get("objectiveConfigSha256"),
                        "objectiveConfigSha256",
                    ),
                    producing_job_id=job_id,
                    created_at=published_at,
                ))
                session.flush()
                alias = session.get(
                    ModelAlias,
                    (job.owner_subject, label),
                    with_for_update=True,
                )
                if alias is None:
                    session.add(ModelAlias(
                        owner_subject=job.owner_subject,
                        label=label,
                        model_ref=model_ref,
                        updated_at=published_at,
                    ))
                else:
                    alias.model_ref = model_ref
                    alias.updated_at = published_at
                record.status = ExecutionState.SUCCEEDED.value
                record.finished_at = published_at
                job.execution_state = ExecutionState.SUCCEEDED.value
                job.revision += 1
                published_result = json_value(result)
                checkpoint = published_result.get("checkpoint")
                if isinstance(checkpoint, dict):
                    checkpoint["generation"] = generation
                job.result = published_result
                job.error_code = None
                job.error_message = None
                job.finished_at = published_at
                job.updated_at = published_at
                session.flush()
                return decode(job)
        except IntegrityError as exc:
            diagnostic = getattr(exc.orig, "diag", None)
            constraint_name = getattr(diagnostic, "constraint_name", None)
            if constraint_name in {
                "models_pkey",
                "models_generation_uq",
                "models_checkpoint_path_uq",
                "models_metadata_path_uq",
                "models_producing_job_id_key",
            }:
                raise conflict("model generation already exists") from exc
            raise RuntimeError(
                "model publication violated persistence invariants"
            ) from exc

    def get_model(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
    ) -> RowMapping | None:
        statement = select(PublishedModel).where(
            PublishedModel.model_ref == model_ref,
            PublishedModel.lifecycle_state
            == ModelLifecycleState.AVAILABLE.value,
        )
        if owner_subject is not None:
            statement = statement.where(
                PublishedModel.owner_subject == owner_subject
            )
        with self.sessions.read(connection) as session:
            return decode_optional(session.scalar(statement))

    def get_model_artifact(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
    ) -> ModelArtifactRecord | None:
        statement = select(PublishedModel).where(
            PublishedModel.model_ref == model_ref,
            PublishedModel.lifecycle_state
            == ModelLifecycleState.AVAILABLE.value,
        )
        if owner_subject is not None:
            statement = statement.where(
                PublishedModel.owner_subject == owner_subject
            )
        with self.sessions.read(connection) as session:
            return _model_artifact_record(session.scalar(statement))

    def get_published_model(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
        for_update: bool = False,
    ) -> PublishedModelRecord | None:
        statement = select(PublishedModel).where(
            PublishedModel.model_ref == model_ref,
            PublishedModel.lifecycle_state
            == ModelLifecycleState.AVAILABLE.value,
        )
        if owner_subject is not None:
            statement = statement.where(
                PublishedModel.owner_subject == owner_subject
            )
        if for_update:
            statement = statement.with_for_update()
        with self.sessions.read(connection) as session:
            return published_model_record(session.scalar(statement))

    def resolve_model_alias(
        self,
        owner_subject: str,
        label: str,
        *,
        connection: Session | None = None,
    ) -> RowMapping | None:
        with self.sessions.read(connection) as session:
            model = session.scalar(
                select(PublishedModel)
                .join(
                    ModelAlias,
                    ModelAlias.model_ref == PublishedModel.model_ref,
                )
                .where(
                    ModelAlias.owner_subject == owner_subject,
                    ModelAlias.label == label,
                    PublishedModel.lifecycle_state
                    == ModelLifecycleState.AVAILABLE.value,
                )
            )
            return decode_optional(model)

    def resolve_published_model_alias(
        self,
        owner_subject: str,
        label: str,
        *,
        connection: Session | None = None,
        for_update: bool = False,
    ) -> PublishedModelRecord | None:
        with self.sessions.read(connection) as session:
            statement = (
                select(PublishedModel)
                .join(
                    ModelAlias,
                    ModelAlias.model_ref == PublishedModel.model_ref,
                )
                .where(
                    ModelAlias.owner_subject == owner_subject,
                    ModelAlias.label == label,
                    PublishedModel.lifecycle_state
                    == ModelLifecycleState.AVAILABLE.value,
                )
            )
            if for_update:
                statement = statement.with_for_update(of=PublishedModel)
            model = session.scalar(statement)
            return published_model_record(model)

    def list_models(self) -> list[RowMapping]:
        with self.database.session() as session:
            rows = session.scalars(
                select(PublishedModel).order_by(
                    PublishedModel.owner_subject,
                    PublishedModel.label,
                    PublishedModel.generation,
                )
            )
            return [decode(row) for row in rows]

    def list_outputs(
        self,
        job_id: str,
        *,
        connection: Session | None = None,
    ) -> list[RowMapping]:
        with self.sessions.read(connection) as session:
            rows = session.scalars(
                select(JobOutput)
                .where(JobOutput.job_id == job_id)
                .order_by(JobOutput.ordinal)
            )
            return [decode(row) for row in rows]

    def list_outputs_page(
        self,
        job_id: str,
        owner_subject: str,
        *,
        cursor: int | None,
        limit: int,
    ) -> RowMapping:
        with self.database.session() as session:
            job = session.scalar(select(Job.job_id).where(
                Job.job_id == job_id,
                Job.owner_subject == owner_subject,
            ))
            if job is None:
                raise not_found("job not found")
            statement = select(JobOutput).where(
                JobOutput.job_id == job_id
            )
            if cursor is not None:
                statement = statement.where(JobOutput.ordinal > cursor)
            rows = list(session.scalars(
                statement.order_by(JobOutput.ordinal).limit(limit + 1)
            ))
        has_more = len(rows) > limit
        page = rows[:limit]
        return {
            "cursor": cursor,
            "items": [decode(row) for row in page],
            "next_cursor": page[-1].ordinal if has_more else None,
            "has_more": has_more,
        }

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
        created_at = timestamp_now(now)
        expires_at = datetime.fromtimestamp(
            created_at.timestamp() + ttl_seconds,
            UTC,
        )
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
                    Job.execution_state == ExecutionState.SUCCEEDED.value,
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
                created_at=created_at,
            ))
        return token, expires_at.timestamp()

    def resolve_ticket(
        self,
        ticket: bytes,
        *,
        owner_subject: str,
        now: float | None = None,
    ) -> RowMapping:
        ticket_hash = hashlib.sha256(bytes(ticket)).hexdigest()
        with self.database.session() as session:
            row = session.execute(
                select(OutputTicket, JobOutput, Job.execution_state)
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
            raise ServiceError(
                ErrorCode.PERMISSION_DENIED,
                "output ticket belongs to another subject",
            )
        if record.expires_at <= timestamp_now(now):
            raise failed_precondition("output ticket has expired")
        if state != ExecutionState.SUCCEEDED.value:
            raise failed_precondition("job output is not available")
        result = decode(output)
        result.update({
            "ticket_owner": record.owner_subject,
            "expires_at": record.expires_at.timestamp(),
            "state": state,
        })
        return result


def _output_record(
    job_id: str,
    output: JsonObject,
    published_at: datetime,
) -> JobOutput:
    relative_path = validate_relative_path(output.get("relative_path"))
    ordinal = nonnegative(output.get("ordinal"), "ordinal")
    rows = nonnegative(output.get("rows"), "rows")
    batches = nonnegative(output.get("batches"), "batches")
    byte_count = nonnegative(output.get("bytes"), "bytes")
    sha256 = digest(output.get("sha256"), "sha256")
    schema_fingerprint = digest(
        output.get("schema_fingerprint"),
        "schema_fingerprint",
    )
    return JobOutput(
        job_id=job_id,
        ordinal=ordinal,
        rows=rows,
        batches=batches,
        bytes=byte_count,
        sha256=sha256,
        schema_fingerprint=schema_fingerprint,
        relative_path=relative_path,
        published_at=published_at,
    )


def _model_artifact_record(
    record: PublishedModel | None,
) -> ModelArtifactRecord | None:
    if record is None:
        return None
    return ModelArtifactRecord(
        model_ref=record.model_ref,
        owner_subject=record.owner_subject,
        checkpoint_path=record.checkpoint_path,
        byte_count=record.checkpoint_bytes,
        sha256=record.sha256,
        data_contract=(
            None
            if record.data_contract is None
            else dict(record.data_contract)
        ),
        ml_contract=(
            None if record.ml_contract is None else dict(record.ml_contract)
        ),
        objective_config_sha256=record.objective_config_sha256,
    )


__all__ = ["ArtifactLedgerSlice"]
