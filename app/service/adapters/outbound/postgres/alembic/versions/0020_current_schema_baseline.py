"""Создать исходную схему PostgreSQL.

Идентификатор ревизии: 0020
Предыдущая ревизия: None

Эта исходная точка заменяет опубликованную цепочку миграций 0001-0020 для
новых баз данных. Базы данных, созданные этой цепочкой, должны уже иметь
ревизию 0020 до использования этой версии исходного кода.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0020"
down_revision = None
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def _metadata(schema: str) -> sa.MetaData:
    metadata = sa.MetaData()
    sa.Sequence(
        "job_queue_sequence_seq",
        metadata=metadata,
        schema=schema,
    )
    _define_job_tables(metadata, schema)
    _define_model_tables(metadata, schema)
    _define_telemetry_tables(metadata, schema)
    _define_control_tables(metadata, schema)
    return metadata


def _define_job_tables(metadata: sa.MetaData, schema: str) -> None:
    sa.Table(
        "job_identities",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
        ),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("create_hash", sa.String(64), nullable=False),
        sa.Column("create_result", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        sa.Index(
            "job_identities_owner_idx",
            "owner_subject",
            "created_at",
        ),
        schema=schema,
    )
    sa.Table(
        "jobs",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
        ),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("operation", sa.String(16), nullable=False),
        sa.Column("input_state", sa.String(16), nullable=False),
        sa.Column("execution_state", sa.String(24), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("input_revision", sa.BigInteger(), nullable=False),
        sa.Column("next_input_ordinal", sa.Integer(), nullable=False),
        sa.Column(
            "client_execution_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("requested_device", sa.String(8), nullable=False),
        sa.Column("selected_device", sa.String(8)),
        sa.Column("model_label", sa.String(256)),
        sa.Column("resolved_model_ref", sa.String(128)),
        sa.Column("prediction_column", sa.String(128), nullable=False),
        sa.Column("model_config", postgresql.JSONB(), nullable=False),
        sa.Column("training_config", postgresql.JSONB()),
        sa.Column("data_contract", postgresql.JSONB(), nullable=False),
        sa.Column("data_contract_sha256", sa.String(64), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("source_width", sa.Integer(), nullable=False),
        sa.Column("feature_dim", sa.Integer(), nullable=False),
        sa.Column("manifest_sha256", sa.String(64)),
        sa.Column(
            "payload_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "total_rows",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "total_bytes",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "progress",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "attempt",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("queue_sequence", sa.BigInteger()),
        sa.Column(
            "waiting_for_input",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("waiting_input_ordinal", sa.Integer()),
        sa.Column("input_waiting_since", sa.DateTime(timezone=True)),
        sa.Column("acquire_grace_until", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text()),
        sa.Column("result", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_closed_at", sa.DateTime(timezone=True)),
        sa.Column("queued_at", sa.DateTime(timezone=True)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("ml_contract", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ("job_id",),
            (f"{schema}.job_identities.job_id",),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "operation IN ('fit', 'predict')",
            name="jobs_operation_ck",
        ),
        sa.CheckConstraint(
            "input_state IN ('OPEN', 'CLOSED', 'ABORTED')",
            name="jobs_input_state_ck",
        ),
        sa.CheckConstraint(
            "execution_state IN ('WAITING_INPUT', 'QUEUED', 'RUNNING', "
            "'RETRYING', 'CANCELLING', 'SUCCEEDED', 'FAILED', 'CANCELLED')",
            name="jobs_execution_state_ck",
        ),
        sa.CheckConstraint("revision >= 1", name="jobs_revision_ck"),
        sa.CheckConstraint(
            "input_revision >= 0",
            name="jobs_input_revision_ck",
        ),
        sa.CheckConstraint(
            "next_input_ordinal >= 0",
            name="jobs_next_ordinal_ck",
        ),
        sa.CheckConstraint(
            "fencing_token >= 1",
            name="jobs_fencing_token_ck",
        ),
        sa.CheckConstraint(
            "requested_device IN ('cpu', 'cuda', 'auto')",
            name="jobs_requested_device_ck",
        ),
        sa.CheckConstraint(
            "selected_device IS NULL OR selected_device IN ('cpu', 'cuda')",
            name="jobs_selected_device_ck",
        ),
        sa.CheckConstraint("source_width > 0", name="jobs_source_width_ck"),
        sa.CheckConstraint("feature_dim > 0", name="jobs_feature_dim_ck"),
        sa.CheckConstraint("attempt >= 0", name="jobs_attempt_ck"),
        sa.CheckConstraint(
            "queue_sequence IS NULL OR queue_sequence > 0",
            name="jobs_queue_sequence_ck",
        ),
        sa.CheckConstraint(
            "payload_count >= 0 AND total_rows >= 0 AND total_bytes >= 0",
            name="jobs_input_totals_ck",
        ),
        sa.CheckConstraint(
            "execution_state <> 'SUCCEEDED' OR input_state = 'CLOSED'",
            name="jobs_success_requires_closed_input_ck",
        ),
        sa.CheckConstraint(
            "(input_state = 'CLOSED' AND manifest_sha256 IS NOT NULL "
            "AND input_closed_at IS NOT NULL) OR "
            "(input_state <> 'CLOSED' AND manifest_sha256 IS NULL "
            "AND input_closed_at IS NULL)",
            name="jobs_closed_manifest_ck",
        ),
        sa.CheckConstraint(
            "(waiting_for_input AND execution_state = 'RUNNING' "
            "AND input_state = 'OPEN' AND waiting_input_ordinal IS NOT NULL "
            "AND input_waiting_since IS NOT NULL) OR "
            "(NOT waiting_for_input AND waiting_input_ordinal IS NULL "
            "AND input_waiting_since IS NULL)",
            name="jobs_input_waiting_ck",
        ),
        sa.CheckConstraint(
            "(operation = 'fit' AND model_label IS NOT NULL "
            "AND resolved_model_ref IS NULL) OR "
            "(operation = 'predict' AND model_label IS NULL "
            "AND resolved_model_ref IS NOT NULL)",
            name="jobs_operation_fields_ck",
        ),
        sa.Index(
            "jobs_queue_idx",
            "execution_state",
            "selected_device",
            "queue_sequence",
        ),
        sa.Index(
            "jobs_owner_state_idx",
            "owner_subject",
            "execution_state",
        ),
        sa.Index(
            "jobs_queue_sequence_idx",
            "queue_sequence",
            unique=True,
            postgresql_where=sa.text("queue_sequence IS NOT NULL"),
        ),
        sa.Index(
            "jobs_input_waiting_idx",
            "waiting_for_input",
            "input_waiting_since",
            postgresql_where=sa.text("waiting_for_input"),
        ),
        schema=schema,
    )
    sa.Table(
        "input_uploads",
        metadata,
        sa.Column("upload_token", sa.String(128), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column(
            "payload_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column(
            "client_execution_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("candidate_path", sa.Text(), nullable=False),
        sa.Column("storage_class", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ("job_id",),
            (f"{schema}.jobs.job_id",),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "job_id",
            "ordinal",
            name="input_uploads_job_ordinal_uq",
        ),
        sa.UniqueConstraint(
            "job_id",
            "payload_id",
            name="input_uploads_job_payload_uq",
        ),
        sa.CheckConstraint(
            "ordinal >= 0",
            name="input_uploads_ordinal_ck",
        ),
        sa.CheckConstraint(
            "fencing_token >= 1",
            name="input_uploads_fencing_token_ck",
        ),
        sa.CheckConstraint(
            "storage_class IN ('runtime', 'recovery')",
            name="input_uploads_storage_class_ck",
        ),
        schema=schema,
    )
    sa.Table(
        "job_inputs",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column(
            "payload_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("commit_revision", sa.BigInteger(), nullable=False),
        sa.Column("schema_id", sa.String(128), nullable=False),
        sa.Column("data_contract_sha256", sa.String(64), nullable=False),
        sa.Column("rows", sa.BigInteger(), nullable=False),
        sa.Column("batches", sa.BigInteger(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("schema_fingerprint", sa.String(64), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("storage_class", sa.String(16), nullable=False),
        sa.Column("source_width", sa.Integer(), nullable=False),
        sa.Column("feature_dim", sa.Integer(), nullable=False),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ("job_id",),
            (f"{schema}.jobs.job_id",),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", "ordinal", name="job_inputs_pk"),
        sa.UniqueConstraint(
            "job_id",
            "payload_id",
            name="job_inputs_job_payload_uq",
        ),
        sa.UniqueConstraint(
            "job_id",
            "commit_revision",
            name="job_inputs_job_revision_uq",
        ),
        sa.UniqueConstraint(
            "relative_path",
            name="job_inputs_relative_path_uq",
        ),
        sa.CheckConstraint("ordinal >= 0", name="job_inputs_ordinal_ck"),
        sa.CheckConstraint(
            "commit_revision > 0",
            name="job_inputs_revision_ck",
        ),
        sa.CheckConstraint("rows >= 0", name="job_inputs_rows_ck"),
        sa.CheckConstraint("batches >= 0", name="job_inputs_batches_ck"),
        sa.CheckConstraint("bytes >= 0", name="job_inputs_bytes_ck"),
        sa.CheckConstraint(
            "source_width > 0",
            name="job_inputs_source_width_ck",
        ),
        sa.CheckConstraint(
            "feature_dim > 0",
            name="job_inputs_feature_dim_ck",
        ),
        sa.CheckConstraint(
            "storage_class IN ('runtime', 'recovery')",
            name="job_inputs_storage_class_ck",
        ),
        sa.Index("job_inputs_revision_idx", "job_id", "commit_revision"),
        schema=schema,
    )
    _define_attempt_tables(metadata, schema)
    _define_job_output(metadata, schema)


def _define_attempt_tables(metadata: sa.MetaData, schema: str) -> None:
    sa.Table(
        "job_attempts",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "attempt_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("selected_device", sa.String(8), nullable=False),
        sa.Column("device_id", sa.String(128)),
        sa.Column("resume_generation", sa.Integer()),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("worker_id", sa.String(256)),
        sa.Column("pid", sa.Integer()),
        sa.Column("pgid", sa.Integer()),
        sa.Column("boot_id", postgresql.UUID(as_uuid=False)),
        sa.Column("process_start_ticks", sa.BigInteger()),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("exit_code", sa.Integer()),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text()),
        sa.Column(
            "queue_entered_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("worker_ready_at", sa.DateTime(timezone=True)),
        sa.Column("worker_completed_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ("job_id",),
            (f"{schema}.jobs.job_id",),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", "attempt", name="job_attempts_pk"),
        sa.UniqueConstraint(
            "attempt_id",
            name="job_attempts_attempt_id_uq",
        ),
        sa.CheckConstraint("attempt > 0", name="job_attempts_attempt_ck"),
        sa.CheckConstraint(
            "selected_device IN ('cpu', 'cuda')",
            name="job_attempts_device_ck",
        ),
        sa.CheckConstraint(
            "status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')",
            name="job_attempts_status_ck",
        ),
        sa.CheckConstraint(
            "process_start_ticks IS NULL OR process_start_ticks > 0",
            name="job_attempts_process_start_ticks_ck",
        ),
        sa.CheckConstraint(
            "resume_generation IS NULL OR resume_generation > 0",
            name="job_attempts_resume_generation_ck",
        ),
        sa.CheckConstraint(
            "(selected_device = 'cpu' AND device_id IS NULL) OR "
            "(selected_device = 'cuda' AND device_id IS NOT NULL)",
            name="job_attempts_device_assignment_ck",
        ),
        schema=schema,
    )
    sa.Table(
        "training_recovery_checkpoints",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("format", sa.String(64), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("completed_epochs", sa.Integer(), nullable=False),
        sa.Column("global_step", sa.BigInteger(), nullable=False),
        sa.Column("training_complete", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ("job_id", "attempt"),
            (f"{schema}.job_attempts.job_id", f"{schema}.job_attempts.attempt"),
            name="training_recovery_checkpoints_attempt_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "job_id",
            "generation",
            name="training_recovery_checkpoints_pk",
        ),
        sa.UniqueConstraint(
            "relative_path",
            name="training_recovery_checkpoints_relative_path_uq",
        ),
        sa.CheckConstraint(
            "generation > 0",
            name="training_recovery_checkpoints_generation_ck",
        ),
        sa.CheckConstraint(
            "attempt > 0",
            name="training_recovery_checkpoints_attempt_ck",
        ),
        sa.CheckConstraint(
            "bytes > 0",
            name="training_recovery_checkpoints_bytes_ck",
        ),
        sa.CheckConstraint(
            "completed_epochs > 0",
            name="training_recovery_checkpoints_epochs_ck",
        ),
        sa.CheckConstraint(
            "global_step >= 0",
            name="training_recovery_checkpoints_step_ck",
        ),
        schema=schema,
    )
    sa.Table(
        "training_metric_intervals",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "attempt_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("checkpoint_serialization_ms", sa.Float()),
        sa.Column("checkpoint_publication_ms", sa.Float()),
        sa.ForeignKeyConstraint(
            ("job_id", "attempt"),
            (f"{schema}.job_attempts.job_id", f"{schema}.job_attempts.attempt"),
            name="training_metric_intervals_attempt_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "job_id",
            "generation",
            name="training_metric_intervals_pk",
        ),
        sa.UniqueConstraint(
            "job_id",
            "attempt_id",
            "generation",
            name="training_metric_intervals_identity_uq",
        ),
        sa.CheckConstraint(
            "generation > 0",
            name="training_metric_intervals_generation_ck",
        ),
        sa.CheckConstraint(
            "attempt > 0",
            name="training_metric_intervals_attempt_ck",
        ),
        sa.CheckConstraint(
            "checkpoint_serialization_ms >= 0",
            name="training_metric_intervals_checkpoint_serialization_ck",
        ),
        sa.CheckConstraint(
            "checkpoint_publication_ms >= 0",
            name="training_metric_intervals_checkpoint_publication_ck",
        ),
        schema=schema,
    )


def _define_job_output(metadata: sa.MetaData, schema: str) -> None:
    sa.Table(
        "job_outputs",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("rows", sa.BigInteger(), nullable=False),
        sa.Column("batches", sa.BigInteger(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("schema_fingerprint", sa.String(64), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ("job_id",),
            (f"{schema}.jobs.job_id",),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", "ordinal", name="job_outputs_pk"),
        sa.UniqueConstraint(
            "relative_path",
            name="job_outputs_relative_path_uq",
        ),
        sa.CheckConstraint("ordinal >= 0", name="job_outputs_ordinal_ck"),
        sa.CheckConstraint("rows >= 0", name="job_outputs_rows_ck"),
        sa.CheckConstraint("batches >= 0", name="job_outputs_batches_ck"),
        sa.CheckConstraint("bytes >= 0", name="job_outputs_bytes_ck"),
        schema=schema,
    )


def _define_model_tables(metadata: sa.MetaData, schema: str) -> None:
    sa.Table(
        "models",
        metadata,
        sa.Column("model_ref", sa.String(128), primary_key=True),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("label", sa.String(256), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("checkpoint_path", sa.Text(), nullable=False),
        sa.Column("metadata_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("producing_job_id", postgresql.UUID(as_uuid=False)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "checkpoint_bytes",
            sa.BigInteger(),
            nullable=False,
            server_default="1",
        ),
        sa.Column("data_contract", postgresql.JSONB()),
        sa.Column("data_contract_sha256", sa.String(64)),
        sa.Column("ml_contract", postgresql.JSONB()),
        sa.Column("objective_config_sha256", sa.String(64)),
        sa.Column(
            "lifecycle_state",
            sa.String(16),
            nullable=False,
            server_default="AVAILABLE",
        ),
        sa.Column("deletion_requested_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ("producing_job_id",),
            (f"{schema}.jobs.job_id",),
            name="models_producing_job_id_fkey",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("producing_job_id"),
        sa.UniqueConstraint(
            "owner_subject",
            "label",
            "generation",
            name="models_generation_uq",
        ),
        sa.UniqueConstraint(
            "checkpoint_path",
            name="models_checkpoint_path_uq",
        ),
        sa.UniqueConstraint(
            "metadata_path",
            name="models_metadata_path_uq",
        ),
        sa.CheckConstraint("generation > 0", name="models_generation_ck"),
        sa.CheckConstraint(
            "checkpoint_bytes > 0",
            name="models_checkpoint_bytes_ck",
        ),
        sa.CheckConstraint(
            "(lifecycle_state = 'AVAILABLE' "
            "AND deletion_requested_at IS NULL) OR "
            "(lifecycle_state = 'DELETING' "
            "AND deletion_requested_at IS NOT NULL)",
            name="models_lifecycle_ck",
        ),
        sa.CheckConstraint(
            "(ml_contract IS NULL AND objective_config_sha256 IS NULL) OR "
            "(ml_contract IS NOT NULL AND objective_config_sha256 IS NOT NULL "
            "AND data_contract IS NOT NULL AND data_contract_sha256 IS NOT NULL)",
            name="models_ml_contract_ck",
        ),
        sa.Index(
            "models_deleting_idx",
            "deletion_requested_at",
            "model_ref",
            postgresql_where=sa.text("lifecycle_state = 'DELETING'"),
        ),
        schema=schema,
    )
    sa.Table(
        "deleted_models",
        metadata,
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("label", sa.String(256), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "deletion_requested_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_ref", name="deleted_models_pk"),
        sa.UniqueConstraint(
            "owner_subject",
            "label",
            "generation",
            name="deleted_models_generation_uq",
        ),
        sa.CheckConstraint(
            "generation > 0",
            name="deleted_models_generation_ck",
        ),
        sa.Index(
            "deleted_models_deleted_at_idx",
            "deleted_at",
            "model_ref",
        ),
        schema=schema,
    )


def _define_telemetry_tables(metadata: sa.MetaData, schema: str) -> None:
    sa.Table(
        "training_metrics_artifacts",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("format", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column(
            "attempt_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("application_version", sa.String(64), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(
            "job_id",
            name="training_metrics_artifacts_pk",
        ),
        sa.UniqueConstraint(
            "model_ref",
            name="training_metrics_artifacts_model_ref_uq",
        ),
        sa.UniqueConstraint(
            "relative_path",
            name="training_metrics_artifacts_relative_path_uq",
        ),
        sa.CheckConstraint(
            "bytes > 0",
            name="training_metrics_artifacts_bytes_ck",
        ),
        sa.CheckConstraint(
            "row_count > 0",
            name="training_metrics_artifacts_rows_ck",
        ),
        sa.CheckConstraint(
            "attempt > 0",
            name="training_metrics_artifacts_attempt_ck",
        ),
        schema=schema,
    )
    sa.Table(
        "fit_run_summary_artifacts",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("format", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column(
            "attempt_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("application_version", sa.String(64), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint(
            "job_id",
            name="fit_run_summary_artifacts_pk",
        ),
        sa.UniqueConstraint(
            "model_ref",
            name="fit_run_summary_artifacts_model_ref_uq",
        ),
        sa.UniqueConstraint(
            "relative_path",
            name="fit_run_summary_artifacts_relative_path_uq",
        ),
        sa.CheckConstraint(
            "bytes > 0",
            name="fit_run_summary_artifacts_bytes_ck",
        ),
        sa.CheckConstraint(
            "attempt > 0",
            name="fit_run_summary_artifacts_attempt_ck",
        ),
        schema=schema,
    )
    sa.Table(
        "metrics_outbox",
        metadata,
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("projection_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("cursor", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error_code", sa.String(64)),
        sa.Column("last_error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ("job_id",),
            (f"{schema}.training_metrics_artifacts.job_id",),
            name="metrics_outbox_artifact_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", name="metrics_outbox_pk"),
        sa.CheckConstraint(
            "status IN ('PENDING', 'BLOCKED', 'DELIVERED', 'CANCELLED')",
            name="metrics_outbox_status_ck",
        ),
        sa.CheckConstraint("cursor >= 0", name="metrics_outbox_cursor_ck"),
        sa.CheckConstraint(
            "attempts >= 0",
            name="metrics_outbox_attempts_ck",
        ),
        sa.Index(
            "metrics_outbox_pending_idx",
            "status",
            "next_attempt_at",
            "created_at",
            postgresql_where=sa.text("status = 'PENDING'"),
        ),
        schema=schema,
    )


def _define_control_tables(metadata: sa.MetaData, schema: str) -> None:
    sa.Table(
        "model_aliases",
        metadata,
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("label", sa.String(256), nullable=False),
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ("model_ref",),
            (f"{schema}.models.model_ref",),
            name="model_aliases_model_ref_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "owner_subject",
            "label",
            name="model_aliases_pk",
        ),
        schema=schema,
    )
    sa.Table(
        "idempotency_records",
        metadata,
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("action_name", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("response", postgresql.JSONB(), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=False)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ("job_id",),
            (f"{schema}.job_identities.job_id",),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint(
            "owner_subject",
            "action_name",
            "idempotency_key",
            name="idempotency_records_pk",
        ),
        schema=schema,
    )
    sa.Table(
        "output_tickets",
        metadata,
        sa.Column("ticket_hash", sa.String(64), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ("job_id", "ordinal"),
            (f"{schema}.job_outputs.job_id", f"{schema}.job_outputs.ordinal"),
            name="output_tickets_output_fk",
            ondelete="CASCADE",
        ),
        sa.Index("output_tickets_expiry_idx", "expires_at"),
        schema=schema,
    )
    sa.Table(
        "api_access_tokens",
        metadata,
        sa.Column(
            "token_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("token_digest", sa.String(64), nullable=False),
        sa.Column("subject", sa.String(256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("token_id", name="api_access_tokens_pk"),
        sa.UniqueConstraint(
            "token_digest",
            name="api_access_tokens_token_digest_uq",
        ),
        sa.CheckConstraint(
            "token_digest ~ '^[0-9a-f]{64}$'",
            name="api_access_tokens_digest_format_ck",
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name="api_access_tokens_expiry_order_ck",
        ),
        sa.Index("api_access_tokens_active_idx", "expires_at"),
        schema=schema,
    )
    sa.Table(
        "runtime_state",
        metadata,
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema=schema,
    )


def upgrade() -> None:
    _metadata(_schema()).create_all(
        bind=op.get_bind(),
        checkfirst=False,
    )


def downgrade() -> None:
    raise RuntimeError(
        "revision 0020 is the PostgreSQL schema baseline and cannot be "
        "downgraded; use the 0.1.15 migration chain for legacy revisions"
    )
