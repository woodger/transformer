from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    Sequence,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


SCHEMA = "transformer"
QUEUE_SEQUENCE = Sequence("job_queue_sequence_seq", schema=SCHEMA)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint("operation IN ('fit', 'predict')", name="jobs_operation_ck"),
        CheckConstraint(
            "state IN ('UPLOADING', 'SEALED', 'QUEUED', 'RUNNING', "
            "'SUCCEEDED', 'FAILED', 'CANCELLING', 'CANCELLED')",
            name="jobs_state_ck",
        ),
        CheckConstraint("revision >= 1", name="jobs_revision_ck"),
        CheckConstraint(
            "requested_device IN ('cpu', 'cuda', 'auto')",
            name="jobs_requested_device_ck",
        ),
        CheckConstraint(
            "selected_device IS NULL OR selected_device IN ('cpu', 'cuda')",
            name="jobs_selected_device_ck",
        ),
        CheckConstraint("source_width IS NULL OR source_width > 0", name="jobs_source_width_ck"),
        CheckConstraint("feature_dim IS NULL OR feature_dim > 0", name="jobs_feature_dim_ck"),
        CheckConstraint("attempt >= 0", name="jobs_attempt_ck"),
        CheckConstraint(
            "queue_sequence IS NULL OR queue_sequence > 0",
            name="jobs_queue_sequence_ck",
        ),
        CheckConstraint(
            "(operation = 'fit' AND model_label IS NOT NULL AND input_model_ref IS NULL) "
            "OR (operation = 'predict' AND model_label IS NULL AND input_model_ref IS NOT NULL)",
            name="jobs_operation_fields_ck",
        ),
        Index("jobs_queue_idx", "state", "selected_device", "queued_at", "job_id"),
        Index("jobs_owner_state_idx", "owner_subject", "state"),
        Index("jobs_queue_claim_idx", "state", "selected_device", "queue_sequence"),
        Index(
            "jobs_queue_sequence_idx",
            "queue_sequence",
            unique=True,
            postgresql_where=text("queue_sequence IS NOT NULL"),
        ),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True)
    owner_subject: Mapped[str] = mapped_column(String(256), nullable=False)
    operation: Mapped[str] = mapped_column(String(16), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    requested_device: Mapped[str] = mapped_column(String(8), nullable=False)
    selected_device: Mapped[str | None] = mapped_column(String(8))
    model_label: Mapped[str | None] = mapped_column(String(256))
    input_model_ref: Mapped[str | None] = mapped_column(String(64))
    prediction_column: Mapped[str] = mapped_column(String(256), nullable=False)
    model_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    training_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_width: Mapped[int | None] = mapped_column(Integer)
    feature_dim: Mapped[int | None] = mapped_column(Integer)
    seal_hash: Mapped[str | None] = mapped_column(String(64))
    seal_manifest: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    seal_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    start_result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    progress: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    queue_sequence: Mapped[int | None] = mapped_column(BigInteger)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class InputUpload(Base):
    __tablename__ = "input_uploads"
    __table_args__ = (
        UniqueConstraint("job_id", "ordinal", name="input_uploads_job_ordinal_uq"),
        UniqueConstraint("job_id", "payload_id", name="input_uploads_job_payload_uq"),
        CheckConstraint("ordinal >= 0", name="input_uploads_ordinal_ck"),
        {"schema": SCHEMA},
    )

    upload_token: Mapped[str] = mapped_column(String(128), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    payload_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    temporary_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JobInput(Base):
    __tablename__ = "job_inputs"
    __table_args__ = (
        PrimaryKeyConstraint("job_id", "ordinal", name="job_inputs_pk"),
        UniqueConstraint("job_id", "payload_id", name="job_inputs_job_payload_uq"),
        UniqueConstraint("relative_path", name="job_inputs_relative_path_uq"),
        CheckConstraint("ordinal >= 0", name="job_inputs_ordinal_ck"),
        CheckConstraint("rows >= 0", name="job_inputs_rows_ck"),
        CheckConstraint("batches >= 0", name="job_inputs_batches_ck"),
        CheckConstraint("bytes >= 0", name="job_inputs_bytes_ck"),
        CheckConstraint("source_width IS NULL OR source_width > 0", name="job_inputs_source_width_ck"),
        CheckConstraint("feature_dim IS NULL OR feature_dim > 0", name="job_inputs_feature_dim_ck"),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="CASCADE")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    payload_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    schema_id: Mapped[str] = mapped_column(String(128), nullable=False)
    rows: Mapped[int] = mapped_column(BigInteger, nullable=False)
    batches: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_width: Mapped[int | None] = mapped_column(Integer)
    feature_dim: Mapped[int | None] = mapped_column(Integer)
    committed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JobAttempt(Base):
    __tablename__ = "job_attempts"
    __table_args__ = (
        PrimaryKeyConstraint("job_id", "attempt", name="job_attempts_pk"),
        CheckConstraint("attempt > 0", name="job_attempts_attempt_ck"),
        CheckConstraint("selected_device IN ('cpu', 'cuda')", name="job_attempts_device_ck"),
        CheckConstraint(
            "status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')",
            name="job_attempts_status_ck",
        ),
        CheckConstraint(
            "process_start_ticks IS NULL OR process_start_ticks > 0",
            name="job_attempts_process_start_ticks_ck",
        ),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="CASCADE")
    )
    attempt: Mapped[int] = mapped_column(Integer)
    selected_device: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(256))
    pid: Mapped[int | None] = mapped_column(Integer)
    pgid: Mapped[int | None] = mapped_column(Integer)
    boot_id: Mapped[str | None] = mapped_column(Uuid(as_uuid=False))
    process_start_ticks: Mapped[int | None] = mapped_column(BigInteger)
    claimed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    exit_code: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class JobOutput(Base):
    __tablename__ = "job_outputs"
    __table_args__ = (
        PrimaryKeyConstraint("job_id", "ordinal", name="job_outputs_pk"),
        UniqueConstraint("relative_path", name="job_outputs_relative_path_uq"),
        CheckConstraint("ordinal >= 0", name="job_outputs_ordinal_ck"),
        CheckConstraint("rows >= 0", name="job_outputs_rows_ck"),
        CheckConstraint("batches >= 0", name="job_outputs_batches_ck"),
        CheckConstraint("bytes >= 0", name="job_outputs_bytes_ck"),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="CASCADE")
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    rows: Mapped[int] = mapped_column(BigInteger, nullable=False)
    batches: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PublishedModel(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("owner_subject", "label", "generation", name="models_generation_uq"),
        UniqueConstraint("checkpoint_path", name="models_checkpoint_path_uq"),
        UniqueConstraint("metadata_path", name="models_metadata_path_uq"),
        CheckConstraint("generation > 0", name="models_generation_ck"),
        {"schema": SCHEMA},
    )

    model_ref: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_subject: Mapped[str] = mapped_column(String(256), nullable=False)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False)
    producing_job_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="SET NULL"),
        unique=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ModelAlias(Base):
    __tablename__ = "model_aliases"
    __table_args__ = (
        PrimaryKeyConstraint("owner_subject", "label", name="model_aliases_pk"),
        {"schema": SCHEMA},
    )

    owner_subject: Mapped[str] = mapped_column(String(256))
    label: Mapped[str] = mapped_column(String(256))
    model_ref: Mapped[str] = mapped_column(
        String(64), ForeignKey(f"{SCHEMA}.models.model_ref", ondelete="CASCADE"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        PrimaryKeyConstraint(
            "owner_subject", "action_name", "idempotency_key", name="idempotency_records_pk"
        ),
        {"schema": SCHEMA},
    )

    owner_subject: Mapped[str] = mapped_column(String(256))
    action_name: Mapped[str] = mapped_column(String(128))
    idempotency_key: Mapped[str] = mapped_column(String(256))
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    job_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False), ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OutputTicket(Base):
    __tablename__ = "output_tickets"
    __table_args__ = (
        ForeignKeyConstraint(
            ("job_id", "ordinal"),
            (f"{SCHEMA}.job_outputs.job_id", f"{SCHEMA}.job_outputs.ordinal"),
            ondelete="CASCADE",
            name="output_tickets_output_fk",
        ),
        Index("output_tickets_expiry_idx", "expires_at"),
        {"schema": SCHEMA},
    )

    ticket_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    owner_subject: Mapped[str] = mapped_column(String(256), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ApiAccessToken(Base):
    __tablename__ = "api_access_tokens"
    __table_args__ = (
        UniqueConstraint("token", name="api_access_tokens_token_uq"),
        CheckConstraint("token ~ '^a\\.[A-Za-z0-9_-]{86}$'", name="api_access_tokens_format_ck"),
        Index("api_access_tokens_active_idx", "revoked_at", postgresql_where=text("revoked_at IS NULL")),
        {"schema": SCHEMA},
    )

    token_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True)
    token: Mapped[str] = mapped_column(String(88), nullable=False)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RuntimeState(Base):
    __tablename__ = "runtime_state"
    __table_args__ = ({"schema": SCHEMA},)

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
