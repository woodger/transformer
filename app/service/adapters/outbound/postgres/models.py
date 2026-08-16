from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
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

from app.contracts.json_types import JsonObject, JsonValue

# SQLAlchemy resolves recursive aliases in postponed Mapped annotations from
# this module's namespace. Keep JsonValue available even though annotations
# refer to it indirectly through JsonObject.
_JSON_VALUE_TYPE = JsonValue

SCHEMA = "transformer"
QUEUE_SEQUENCE = Sequence("job_queue_sequence_seq", schema=SCHEMA)


class Base(DeclarativeBase):
    pass


class JobIdentity(Base):
    __tablename__ = "job_identities"
    __table_args__ = (
        Index("job_identities_owner_idx", "owner_subject", "created_at"),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True)
    owner_subject: Mapped[str] = mapped_column(String(256), nullable=False)
    create_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    create_result: Mapped[JsonObject] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint("operation IN ('fit', 'predict')", name="jobs_operation_ck"),
        CheckConstraint(
            "input_state IN ('OPEN', 'CLOSED', 'ABORTED')",
            name="jobs_input_state_ck",
        ),
        CheckConstraint(
            "execution_state IN ('WAITING_INPUT', 'QUEUED', 'RUNNING', "
            "'RETRYING', 'CANCELLING', 'SUCCEEDED', 'FAILED', 'CANCELLED')",
            name="jobs_execution_state_ck",
        ),
        CheckConstraint("revision >= 1", name="jobs_revision_ck"),
        CheckConstraint("input_revision >= 0", name="jobs_input_revision_ck"),
        CheckConstraint("next_input_ordinal >= 0", name="jobs_next_ordinal_ck"),
        CheckConstraint("fencing_token >= 1", name="jobs_fencing_token_ck"),
        CheckConstraint(
            "requested_device IN ('cpu', 'cuda', 'auto')",
            name="jobs_requested_device_ck",
        ),
        CheckConstraint(
            "selected_device IS NULL OR selected_device IN ('cpu', 'cuda')",
            name="jobs_selected_device_ck",
        ),
        CheckConstraint("source_width > 0", name="jobs_source_width_ck"),
        CheckConstraint("feature_dim > 0", name="jobs_feature_dim_ck"),
        CheckConstraint("attempt >= 0", name="jobs_attempt_ck"),
        CheckConstraint(
            "queue_sequence IS NULL OR queue_sequence > 0",
            name="jobs_queue_sequence_ck",
        ),
        CheckConstraint(
            "payload_count >= 0 AND total_rows >= 0 AND total_bytes >= 0",
            name="jobs_input_totals_ck",
        ),
        CheckConstraint(
            "execution_state <> 'SUCCEEDED' OR input_state = 'CLOSED'",
            name="jobs_success_requires_closed_input_ck",
        ),
        CheckConstraint(
            "(input_state = 'CLOSED' AND manifest_sha256 IS NOT NULL "
            "AND input_closed_at IS NOT NULL) OR "
            "(input_state <> 'CLOSED' AND manifest_sha256 IS NULL "
            "AND input_closed_at IS NULL)",
            name="jobs_closed_manifest_ck",
        ),
        CheckConstraint(
            "(waiting_for_input AND execution_state = 'RUNNING' "
            "AND input_state = 'OPEN' AND waiting_input_ordinal IS NOT NULL "
            "AND input_waiting_since IS NOT NULL) OR "
            "(NOT waiting_for_input AND waiting_input_ordinal IS NULL "
            "AND input_waiting_since IS NULL)",
            name="jobs_input_waiting_ck",
        ),
        CheckConstraint(
            "(operation = 'fit' AND model_label IS NOT NULL "
            "AND resolved_model_ref IS NULL) OR "
            "(operation = 'predict' AND model_label IS NULL "
            "AND resolved_model_ref IS NOT NULL)",
            name="jobs_operation_fields_ck",
        ),
        Index(
            "jobs_queue_idx",
            "execution_state",
            "selected_device",
            "queue_sequence",
        ),
        Index("jobs_owner_state_idx", "owner_subject", "execution_state"),
        Index(
            "jobs_queue_sequence_idx",
            "queue_sequence",
            unique=True,
            postgresql_where=text("queue_sequence IS NOT NULL"),
        ),
        Index(
            "jobs_input_waiting_idx",
            "waiting_for_input",
            "input_waiting_since",
            postgresql_where=text("waiting_for_input"),
        ),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey(f"{SCHEMA}.job_identities.job_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    owner_subject: Mapped[str] = mapped_column(String(256), nullable=False)
    operation: Mapped[str] = mapped_column(String(16), nullable=False)
    input_state: Mapped[str] = mapped_column(String(16), nullable=False)
    execution_state: Mapped[str] = mapped_column(String(24), nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    input_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    next_input_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    client_execution_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False)
    requested_device: Mapped[str] = mapped_column(String(8), nullable=False)
    selected_device: Mapped[str | None] = mapped_column(String(8))
    model_label: Mapped[str | None] = mapped_column(String(256))
    resolved_model_ref: Mapped[str | None] = mapped_column(String(128))
    prediction_column: Mapped[str] = mapped_column(String(128), nullable=False)
    model_config: Mapped[JsonObject] = mapped_column(JSONB, nullable=False)
    training_config: Mapped[JsonObject | None] = mapped_column(JSONB)
    data_contract: Mapped[JsonObject] = mapped_column(JSONB, nullable=False)
    data_contract_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    ml_contract: Mapped[JsonObject] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_width: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    payload_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_rows: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    total_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    progress: Mapped[JsonObject] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    queue_sequence: Mapped[int | None] = mapped_column(BigInteger)
    waiting_for_input: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    waiting_input_ordinal: Mapped[int | None] = mapped_column(Integer)
    input_waiting_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acquire_grace_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    result: Mapped[JsonObject | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    input_closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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
        CheckConstraint("fencing_token >= 1", name="input_uploads_fencing_token_ck"),
        CheckConstraint(
            "storage_class IN ('runtime', 'recovery')",
            name="input_uploads_storage_class_ck",
        ),
        {"schema": SCHEMA},
    )

    upload_token: Mapped[str] = mapped_column(String(128), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    payload_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    client_execution_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False)
    candidate_path: Mapped[str] = mapped_column(Text, nullable=False)
    storage_class: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JobInput(Base):
    __tablename__ = "job_inputs"
    __table_args__ = (
        PrimaryKeyConstraint("job_id", "ordinal", name="job_inputs_pk"),
        UniqueConstraint("job_id", "payload_id", name="job_inputs_job_payload_uq"),
        UniqueConstraint("job_id", "commit_revision", name="job_inputs_job_revision_uq"),
        UniqueConstraint("relative_path", name="job_inputs_relative_path_uq"),
        CheckConstraint("ordinal >= 0", name="job_inputs_ordinal_ck"),
        CheckConstraint("commit_revision > 0", name="job_inputs_revision_ck"),
        CheckConstraint("rows >= 0", name="job_inputs_rows_ck"),
        CheckConstraint("batches >= 0", name="job_inputs_batches_ck"),
        CheckConstraint("bytes >= 0", name="job_inputs_bytes_ck"),
        CheckConstraint("source_width > 0", name="job_inputs_source_width_ck"),
        CheckConstraint("feature_dim > 0", name="job_inputs_feature_dim_ck"),
        CheckConstraint(
            "storage_class IN ('runtime', 'recovery')",
            name="job_inputs_storage_class_ck",
        ),
        Index("job_inputs_revision_idx", "job_id", "commit_revision"),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="CASCADE"),
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    payload_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    commit_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_id: Mapped[str] = mapped_column(String(128), nullable=False)
    data_contract_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    rows: Mapped[int] = mapped_column(BigInteger, nullable=False)
    batches: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    storage_class: Mapped[str] = mapped_column(String(16), nullable=False)
    source_width: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    committed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JobAttempt(Base):
    __tablename__ = "job_attempts"
    __table_args__ = (
        PrimaryKeyConstraint("job_id", "attempt", name="job_attempts_pk"),
        UniqueConstraint("attempt_id", name="job_attempts_attempt_id_uq"),
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
        CheckConstraint(
            "resume_generation IS NULL OR resume_generation > 0",
            name="job_attempts_resume_generation_ck",
        ),
        CheckConstraint(
            "(selected_device = 'cpu' AND device_id IS NULL) OR "
            "(selected_device = 'cuda' AND device_id IS NOT NULL)",
            name="job_attempts_device_assignment_ck",
        ),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="CASCADE"),
    )
    attempt: Mapped[int] = mapped_column(Integer)
    attempt_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    selected_device: Mapped[str] = mapped_column(String(8), nullable=False)
    device_id: Mapped[str | None] = mapped_column(String(128))
    resume_generation: Mapped[int | None] = mapped_column(Integer)
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


class TrainingRecoveryCheckpoint(Base):
    __tablename__ = "training_recovery_checkpoints"
    __table_args__ = (
        PrimaryKeyConstraint("job_id", "generation", name="training_recovery_checkpoints_pk"),
        ForeignKeyConstraint(
            ("job_id", "attempt"),
            (f"{SCHEMA}.job_attempts.job_id", f"{SCHEMA}.job_attempts.attempt"),
            ondelete="CASCADE",
            name="training_recovery_checkpoints_attempt_fk",
        ),
        UniqueConstraint("relative_path", name="training_recovery_checkpoints_relative_path_uq"),
        CheckConstraint("generation > 0", name="training_recovery_checkpoints_generation_ck"),
        CheckConstraint("attempt > 0", name="training_recovery_checkpoints_attempt_ck"),
        CheckConstraint("bytes > 0", name="training_recovery_checkpoints_bytes_ck"),
        CheckConstraint("completed_epochs > 0", name="training_recovery_checkpoints_epochs_ck"),
        CheckConstraint("global_step >= 0", name="training_recovery_checkpoints_step_ck"),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(Uuid(as_uuid=False))
    generation: Mapped[int] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    format: Mapped[str] = mapped_column(String(64), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    completed_epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    global_step: Mapped[int] = mapped_column(BigInteger, nullable=False)
    training_complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TrainingMetricInterval(Base):
    __tablename__ = "training_metric_intervals"
    __table_args__ = (
        PrimaryKeyConstraint(
            "job_id",
            "generation",
            name="training_metric_intervals_pk",
        ),
        ForeignKeyConstraint(
            ("job_id", "attempt"),
            (f"{SCHEMA}.job_attempts.job_id", f"{SCHEMA}.job_attempts.attempt"),
            ondelete="CASCADE",
            name="training_metric_intervals_attempt_fk",
        ),
        CheckConstraint(
            "generation > 0",
            name="training_metric_intervals_generation_ck",
        ),
        CheckConstraint(
            "attempt > 0",
            name="training_metric_intervals_attempt_ck",
        ),
        UniqueConstraint(
            "job_id",
            "attempt_id",
            "generation",
            name="training_metric_intervals_identity_uq",
        ),
        {"schema": SCHEMA},
    )

    job_id: Mapped[str] = mapped_column(Uuid(as_uuid=False))
    generation: Mapped[int] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    metrics: Mapped[JsonObject] = mapped_column(JSONB, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


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
        Uuid(as_uuid=False),
        ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="CASCADE"),
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
        CheckConstraint("checkpoint_bytes > 0", name="models_checkpoint_bytes_ck"),
        CheckConstraint(
            "(lifecycle_state = 'AVAILABLE' "
            "AND deletion_requested_at IS NULL AND deleted_at IS NULL) OR "
            "(lifecycle_state = 'DELETING' "
            "AND deletion_requested_at IS NOT NULL AND deleted_at IS NULL) OR "
            "(lifecycle_state = 'DELETED' "
            "AND deletion_requested_at IS NOT NULL AND deleted_at IS NOT NULL)",
            name="models_lifecycle_ck",
        ),
        CheckConstraint(
            "(ml_contract IS NULL AND objective_config_sha256 IS NULL) OR "
            "(ml_contract IS NOT NULL AND objective_config_sha256 IS NOT NULL "
            "AND data_contract IS NOT NULL AND data_contract_sha256 IS NOT NULL)",
            name="models_ml_contract_ck",
        ),
        Index(
            "models_deleting_idx",
            "deletion_requested_at",
            "model_ref",
            postgresql_where=text("lifecycle_state = 'DELETING'"),
        ),
        {"schema": SCHEMA},
    )

    model_ref: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_subject: Mapped[str] = mapped_column(String(256), nullable=False)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_path: Mapped[str] = mapped_column(Text, nullable=False)
    checkpoint_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[JsonObject] = mapped_column("metadata", JSONB, nullable=False)
    data_contract: Mapped[JsonObject | None] = mapped_column(JSONB)
    data_contract_sha256: Mapped[str | None] = mapped_column(String(64))
    ml_contract: Mapped[JsonObject | None] = mapped_column(JSONB)
    objective_config_sha256: Mapped[str | None] = mapped_column(String(64))
    producing_job_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey(f"{SCHEMA}.jobs.job_id", ondelete="SET NULL"),
        unique=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lifecycle_state: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="AVAILABLE",
        server_default="AVAILABLE",
    )
    deletion_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelMetricsArtifact(Base):
    __tablename__ = "model_metrics_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "relative_path",
            name="model_metrics_artifacts_relative_path_uq",
        ),
        CheckConstraint(
            "bytes > 0",
            name="model_metrics_artifacts_bytes_ck",
        ),
        CheckConstraint(
            "row_count > 0",
            name="model_metrics_artifacts_rows_ck",
        ),
        CheckConstraint(
            "attempt > 0",
            name="model_metrics_artifacts_attempt_ck",
        ),
        {"schema": SCHEMA},
    )

    model_ref: Mapped[str] = mapped_column(
        String(128),
        ForeignKey(f"{SCHEMA}.models.model_ref", ondelete="CASCADE"),
        primary_key=True,
    )
    format: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    job_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    attempt_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    application_version: Mapped[str] = mapped_column(String(64), nullable=False)
    git_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class MetricsOutboxEntry(Base):
    __tablename__ = "metrics_outbox"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'BLOCKED', 'DELIVERED', 'CANCELLED')",
            name="metrics_outbox_status_ck",
        ),
        CheckConstraint("cursor >= 0", name="metrics_outbox_cursor_ck"),
        CheckConstraint("attempts >= 0", name="metrics_outbox_attempts_ck"),
        Index(
            "metrics_outbox_pending_idx",
            "status",
            "next_attempt_at",
            "created_at",
            postgresql_where=text("status = 'PENDING'"),
        ),
        {"schema": SCHEMA},
    )

    model_ref: Mapped[str] = mapped_column(
        String(128),
        ForeignKey(
            f"{SCHEMA}.model_metrics_artifacts.model_ref",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    projection_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    cursor: Mapped[int] = mapped_column(Integer, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    last_error_code: Mapped[str | None] = mapped_column(String(64))
    last_error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class ModelAlias(Base):
    __tablename__ = "model_aliases"
    __table_args__ = (
        PrimaryKeyConstraint("owner_subject", "label", name="model_aliases_pk"),
        {"schema": SCHEMA},
    )

    owner_subject: Mapped[str] = mapped_column(String(256))
    label: Mapped[str] = mapped_column(String(256))
    model_ref: Mapped[str] = mapped_column(
        String(128),
        ForeignKey(f"{SCHEMA}.models.model_ref", ondelete="CASCADE"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        PrimaryKeyConstraint(
            "owner_subject",
            "action_name",
            "idempotency_key",
            name="idempotency_records_pk",
        ),
        {"schema": SCHEMA},
    )

    owner_subject: Mapped[str] = mapped_column(String(256))
    action_name: Mapped[str] = mapped_column(String(128))
    idempotency_key: Mapped[str] = mapped_column(String(256))
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response: Mapped[JsonObject] = mapped_column(JSONB, nullable=False)
    job_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey(f"{SCHEMA}.job_identities.job_id", ondelete="SET NULL"),
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
        CheckConstraint(
            "token ~ '^a\\.[A-Za-z0-9_-]{86}$'",
            name="api_access_tokens_format_ck",
        ),
        Index(
            "api_access_tokens_active_idx",
            "revoked_at",
            postgresql_where=text("revoked_at IS NULL"),
        ),
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
