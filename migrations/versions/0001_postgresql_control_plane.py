"""Create the PostgreSQL control plane.

Revision ID: 0001
Revises: None
"""
from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.execute(sa.schema.CreateSequence(sa.Sequence("job_queue_sequence_seq", schema=schema)))

    op.create_table(
        "jobs",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("operation", sa.String(16), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("requested_device", sa.String(8), nullable=False),
        sa.Column("selected_device", sa.String(8)),
        sa.Column("model_label", sa.String(256)),
        sa.Column("input_model_ref", sa.String(64)),
        sa.Column("prediction_column", sa.String(256), nullable=False),
        sa.Column("model_config", postgresql.JSONB()),
        sa.Column("training_config", postgresql.JSONB()),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("source_width", sa.Integer()),
        sa.Column("feature_dim", sa.Integer()),
        sa.Column("seal_hash", sa.String(64)),
        sa.Column("seal_manifest", postgresql.JSONB()),
        sa.Column("seal_result", postgresql.JSONB()),
        sa.Column("start_result", postgresql.JSONB()),
        sa.Column("progress", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("queue_sequence", sa.BigInteger()),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text()),
        sa.Column("result", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sealed_at", sa.DateTime(timezone=True)),
        sa.Column("queued_at", sa.DateTime(timezone=True)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("operation IN ('fit', 'predict')", name="jobs_operation_ck"),
        sa.CheckConstraint(
            "state IN ('UPLOADING', 'SEALED', 'QUEUED', 'RUNNING', 'SUCCEEDED', "
            "'FAILED', 'CANCELLING', 'CANCELLED')",
            name="jobs_state_ck",
        ),
        sa.CheckConstraint("revision >= 1", name="jobs_revision_ck"),
        sa.CheckConstraint("requested_device IN ('cpu', 'cuda', 'auto')", name="jobs_requested_device_ck"),
        sa.CheckConstraint("selected_device IS NULL OR selected_device IN ('cpu', 'cuda')", name="jobs_selected_device_ck"),
        sa.CheckConstraint("source_width IS NULL OR source_width > 0", name="jobs_source_width_ck"),
        sa.CheckConstraint("feature_dim IS NULL OR feature_dim > 0", name="jobs_feature_dim_ck"),
        sa.CheckConstraint("attempt >= 0", name="jobs_attempt_ck"),
        sa.CheckConstraint("queue_sequence IS NULL OR queue_sequence > 0", name="jobs_queue_sequence_ck"),
        sa.CheckConstraint(
            "(operation = 'fit' AND model_label IS NOT NULL AND input_model_ref IS NULL) "
            "OR (operation = 'predict' AND model_label IS NULL AND input_model_ref IS NOT NULL)",
            name="jobs_operation_fields_ck",
        ),
        schema=schema,
    )
    op.create_index("jobs_queue_idx", "jobs", ["state", "selected_device", "queued_at", "job_id"], schema=schema)
    op.create_index("jobs_owner_state_idx", "jobs", ["owner_subject", "state"], schema=schema)
    op.create_index("jobs_queue_claim_idx", "jobs", ["state", "selected_device", "queue_sequence"], schema=schema)
    op.create_index(
        "jobs_queue_sequence_idx",
        "jobs",
        ["queue_sequence"],
        unique=True,
        schema=schema,
        postgresql_where=sa.text("queue_sequence IS NOT NULL"),
    )

    op.create_table(
        "input_uploads",
        sa.Column("upload_token", sa.String(128), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("payload_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("temporary_path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], [f"{schema}.jobs.job_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("job_id", "ordinal", name="input_uploads_job_ordinal_uq"),
        sa.UniqueConstraint("job_id", "payload_id", name="input_uploads_job_payload_uq"),
        sa.CheckConstraint("ordinal >= 0", name="input_uploads_ordinal_ck"),
        schema=schema,
    )
    op.create_table(
        "job_inputs",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("payload_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("schema_id", sa.String(128), nullable=False),
        sa.Column("rows", sa.BigInteger(), nullable=False),
        sa.Column("batches", sa.BigInteger(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("schema_fingerprint", sa.String(64), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("source_width", sa.Integer()),
        sa.Column("feature_dim", sa.Integer()),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], [f"{schema}.jobs.job_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id", "ordinal", name="job_inputs_pk"),
        sa.UniqueConstraint("job_id", "payload_id", name="job_inputs_job_payload_uq"),
        sa.UniqueConstraint("relative_path", name="job_inputs_relative_path_uq"),
        sa.CheckConstraint("ordinal >= 0", name="job_inputs_ordinal_ck"),
        sa.CheckConstraint("rows >= 0", name="job_inputs_rows_ck"),
        sa.CheckConstraint("batches >= 0", name="job_inputs_batches_ck"),
        sa.CheckConstraint("bytes >= 0", name="job_inputs_bytes_ck"),
        sa.CheckConstraint("source_width IS NULL OR source_width > 0", name="job_inputs_source_width_ck"),
        sa.CheckConstraint("feature_dim IS NULL OR feature_dim > 0", name="job_inputs_feature_dim_ck"),
        schema=schema,
    )
    op.create_table(
        "job_attempts",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("selected_device", sa.String(8), nullable=False),
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
        sa.ForeignKeyConstraint(["job_id"], [f"{schema}.jobs.job_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id", "attempt", name="job_attempts_pk"),
        sa.CheckConstraint("attempt > 0", name="job_attempts_attempt_ck"),
        sa.CheckConstraint("selected_device IN ('cpu', 'cuda')", name="job_attempts_device_ck"),
        sa.CheckConstraint("status IN ('RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED')", name="job_attempts_status_ck"),
        sa.CheckConstraint("process_start_ticks IS NULL OR process_start_ticks > 0", name="job_attempts_process_start_ticks_ck"),
        schema=schema,
    )
    op.create_table(
        "job_outputs",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("rows", sa.BigInteger(), nullable=False),
        sa.Column("batches", sa.BigInteger(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("schema_fingerprint", sa.String(64), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], [f"{schema}.jobs.job_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id", "ordinal", name="job_outputs_pk"),
        sa.UniqueConstraint("relative_path", name="job_outputs_relative_path_uq"),
        sa.CheckConstraint("ordinal >= 0", name="job_outputs_ordinal_ck"),
        sa.CheckConstraint("rows >= 0", name="job_outputs_rows_ck"),
        sa.CheckConstraint("batches >= 0", name="job_outputs_batches_ck"),
        sa.CheckConstraint("bytes >= 0", name="job_outputs_bytes_ck"),
        schema=schema,
    )
    op.create_table(
        "models",
        sa.Column("model_ref", sa.String(64), primary_key=True),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("label", sa.String(256), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("checkpoint_path", sa.Text(), nullable=False),
        sa.Column("metadata_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("producing_job_id", postgresql.UUID(as_uuid=False)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["producing_job_id"], [f"{schema}.jobs.job_id"], ondelete="SET NULL"),
        sa.UniqueConstraint("producing_job_id"),
        sa.UniqueConstraint("owner_subject", "label", "generation", name="models_generation_uq"),
        sa.UniqueConstraint("checkpoint_path", name="models_checkpoint_path_uq"),
        sa.UniqueConstraint("metadata_path", name="models_metadata_path_uq"),
        sa.CheckConstraint("generation > 0", name="models_generation_ck"),
        schema=schema,
    )
    op.create_table(
        "model_aliases",
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("label", sa.String(256), nullable=False),
        sa.Column("model_ref", sa.String(64), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_ref"], [f"{schema}.models.model_ref"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("owner_subject", "label", name="model_aliases_pk"),
        schema=schema,
    )
    op.create_table(
        "idempotency_records",
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("action_name", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("response", postgresql.JSONB(), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=False)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], [f"{schema}.jobs.job_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("owner_subject", "action_name", "idempotency_key", name="idempotency_records_pk"),
        schema=schema,
    )
    op.create_table(
        "output_tickets",
        sa.Column("ticket_hash", sa.String(64), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id", "ordinal"],
            [f"{schema}.job_outputs.job_id", f"{schema}.job_outputs.ordinal"],
            ondelete="CASCADE",
            name="output_tickets_output_fk",
        ),
        schema=schema,
    )
    op.create_index("output_tickets_expiry_idx", "output_tickets", ["expires_at"], schema=schema)
    op.create_table(
        "api_access_tokens",
        sa.Column("token_id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("token", sa.String(88), nullable=False),
        sa.Column("subject", sa.String(256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("token", name="api_access_tokens_token_uq"),
        sa.CheckConstraint("token ~ '^a\\.[A-Za-z0-9_-]{86}$'", name="api_access_tokens_format_ck"),
        schema=schema,
    )
    op.create_index(
        "api_access_tokens_active_idx",
        "api_access_tokens",
        ["revoked_at"],
        schema=schema,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.create_table(
        "runtime_state",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        schema=schema,
    )

    quoted = schema.replace('"', '""')
    op.execute(sa.text(
        f'CREATE FUNCTION "{quoted}".notify_auth_token_change() RETURNS trigger '
        "LANGUAGE plpgsql AS $$ BEGIN "
        "PERFORM pg_notify('transformer_auth_tokens', 'changed'); "
        "RETURN COALESCE(NEW, OLD); END; $$"
    ))
    op.execute(sa.text(
        f'CREATE TRIGGER api_access_tokens_notify AFTER INSERT OR UPDATE OR DELETE '
        f'ON "{quoted}".api_access_tokens FOR EACH STATEMENT '
        f'EXECUTE FUNCTION "{quoted}".notify_auth_token_change()'
    ))


def downgrade() -> None:
    schema = _schema()
    quoted = schema.replace('"', '""')
    op.execute(sa.text(f'DROP TRIGGER IF EXISTS api_access_tokens_notify ON "{quoted}".api_access_tokens'))
    op.execute(sa.text(f'DROP FUNCTION IF EXISTS "{quoted}".notify_auth_token_change()'))
    for table in (
        "runtime_state",
        "api_access_tokens",
        "output_tickets",
        "idempotency_records",
        "model_aliases",
        "models",
        "job_outputs",
        "job_attempts",
        "job_inputs",
        "input_uploads",
        "jobs",
    ):
        op.drop_table(table, schema=schema)
    op.execute(sa.schema.DropSequence(sa.Sequence("job_queue_sequence_seq", schema=schema)))
