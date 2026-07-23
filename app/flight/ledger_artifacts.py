from __future__ import annotations

from datetime import datetime, UTC
import hashlib
import secrets
from collections.abc import Sequence

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import (
    Job,
    JobAttempt,
    JobOutput,
    ModelAlias,
    OutputTicket,
    PublishedModel,
)
from app.flight.constants import ErrorCode, JobState
from app.flight.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    not_found,
)
from app.flight.ledger_support import (
    advisory_lock,
    decode,
    digest,
    json_value,
    nonnegative,
    now as timestamp_now,
    validate_relative_path,
)
from app.flight.records import ModelArtifactRecord


class ArtifactLedgerSlice:
    """Atomic publication and retrieval of Transformer-owned artifacts."""

    def __init__(self, sessions):
        self.sessions = sessions
        self.database = sessions.database

    def publish_outputs(
        self,
        job_id: str,
        attempt: int,
        outputs: Sequence[dict],
        *,
        result: dict,
        now: float | None = None,
    ) -> dict:
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
                    job.operation != "predict"
                    or job.state != JobState.RUNNING.value
                    or job.attempt != attempt
                ):
                    raise failed_precondition(
                        "job is not the active running attempt"
                    )
                for output in outputs:
                    session.add(
                        _output_record(job_id, output, published_at)
                    )
                session.flush()
                record = session.get(
                    JobAttempt,
                    (job_id, attempt),
                    with_for_update=True,
                )
                record.status = JobState.SUCCEEDED.value
                record.finished_at = published_at
                job.state = JobState.SUCCEEDED.value
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
        validate_relative_path(checkpoint_path)
        validate_relative_path(metadata_path)
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
                    or job.state != JobState.RUNNING.value
                    or job.attempt != attempt
                ):
                    raise failed_precondition(
                        "job is not the active fit attempt"
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
                next_generation = int(session.scalar(
                    select(
                        func.coalesce(
                            func.max(PublishedModel.generation),
                            0,
                        )
                        + 1
                    ).where(
                        PublishedModel.owner_subject == job.owner_subject,
                        PublishedModel.label == label,
                    )
                ))
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
                    sha256=sha256,
                    metadata_json=json_value(metadata),
                    producing_job_id=job_id,
                    created_at=published_at,
                ))
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
                record = session.get(
                    JobAttempt,
                    (job_id, attempt),
                    with_for_update=True,
                )
                record.status = JobState.SUCCEEDED.value
                record.finished_at = published_at
                job.state = JobState.SUCCEEDED.value
                job.revision += 1
                job.result = json_value(result)
                job.error_code = None
                job.error_message = None
                job.finished_at = published_at
                job.updated_at = published_at
                session.flush()
                return decode(job)
        except IntegrityError as exc:
            raise conflict("model generation already exists") from exc

    def get_model(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
    ) -> dict | None:
        statement = select(PublishedModel).where(
            PublishedModel.model_ref == model_ref
        )
        if owner_subject is not None:
            statement = statement.where(
                PublishedModel.owner_subject == owner_subject
            )
        with self.sessions.read(connection) as session:
            return decode(session.scalar(statement))

    def get_model_artifact(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
        connection: Session | None = None,
    ) -> ModelArtifactRecord | None:
        statement = select(PublishedModel).where(
            PublishedModel.model_ref == model_ref
        )
        if owner_subject is not None:
            statement = statement.where(
                PublishedModel.owner_subject == owner_subject
            )
        with self.sessions.read(connection) as session:
            return _model_artifact_record(session.scalar(statement))

    def resolve_model_alias(
        self,
        owner_subject: str,
        label: str,
        *,
        connection: Session | None = None,
    ) -> dict | None:
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
                )
            )
            return decode(model)

    def list_models(self) -> list[dict]:
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
    ) -> list[dict]:
        with self.sessions.read(connection) as session:
            rows = session.scalars(
                select(JobOutput)
                .where(JobOutput.job_id == job_id)
                .order_by(JobOutput.ordinal)
            )
            return [decode(row) for row in rows]

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
                created_at=created_at,
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
            raise ServiceError(
                ErrorCode.PERMISSION_DENIED,
                "output ticket belongs to another subject",
            )
        if record.expires_at <= timestamp_now(now):
            raise failed_precondition("output ticket has expired")
        if state != JobState.SUCCEEDED.value:
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
    output: dict,
    published_at: datetime,
) -> JobOutput:
    relative_path = output["relative_path"]
    validate_relative_path(relative_path)
    for name in ("ordinal", "rows", "batches", "bytes"):
        nonnegative(output[name], name)
    digest(output["sha256"], "sha256")
    digest(output["schema_fingerprint"], "schema_fingerprint")
    return JobOutput(
        job_id=job_id,
        ordinal=output["ordinal"],
        rows=output["rows"],
        batches=output["batches"],
        bytes=output["bytes"],
        sha256=output["sha256"],
        schema_fingerprint=output["schema_fingerprint"],
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
        sha256=record.sha256,
    )


__all__ = ["ArtifactLedgerSlice"]
