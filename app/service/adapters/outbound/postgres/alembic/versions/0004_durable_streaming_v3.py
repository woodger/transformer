"""Replace Flight v2 job state with durable streaming v3.

Revision ID: 0004
Revises: 0003
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = schema.replace('"', '""')

    # V2 runtime state has no valid representation in v3. API tokens and
    # published model identities are deliberately outside this destructive
    # boundary. Legacy models remain present but uncertified for v3.
    op.execute(sa.text(f'UPDATE "{quoted}".models SET producing_job_id = NULL'))
    for table in (
        "output_tickets",
        "training_recovery_checkpoints",
        "job_outputs",
        "job_attempts",
        "job_inputs",
        "input_uploads",
        "idempotency_records",
        "jobs",
    ):
        op.execute(sa.text(f'DROP TABLE "{quoted}"."{table}" CASCADE'))
    op.execute(sa.text(
        f'DROP SEQUENCE IF EXISTS "{quoted}".job_queue_sequence_seq'
    ))
    op.execute(sa.text(
        f'DELETE FROM "{quoted}".runtime_state WHERE key = \'storage_epoch\''
    ))

    op.add_column(
        "models",
        sa.Column(
            "checkpoint_bytes",
            sa.BigInteger(),
            nullable=False,
            server_default="1",
        ),
        schema=schema,
    )
    op.add_column(
        "models",
        sa.Column("data_contract", postgresql.JSONB()),
        schema=schema,
    )
    op.add_column(
        "models",
        sa.Column("data_contract_sha256", sa.String(64)),
        schema=schema,
    )
    op.add_column(
        "models",
        sa.Column(
            "certified_for_v3",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        schema=schema,
    )
    op.create_check_constraint(
        "models_checkpoint_bytes_ck",
        "models",
        "checkpoint_bytes > 0",
        schema=schema,
    )
    op.create_check_constraint(
        "models_v3_certification_ck",
        "models",
        "(certified_for_v3 AND data_contract IS NOT NULL "
        "AND data_contract_sha256 IS NOT NULL) OR NOT certified_for_v3",
        schema=schema,
    )

    op.execute(sa.schema.CreateSequence(
        sa.Sequence("job_queue_sequence_seq", schema=schema)
    ))
    op.create_table(
        "job_identities",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("create_hash", sa.String(64), nullable=False),
        sa.Column("create_result", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        schema=schema,
    )
    op.create_index(
        "job_identities_owner_idx",
        "job_identities",
        ["owner_subject", "created_at"],
        schema=schema,
    )

    op.create_table(
        "jobs",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("operation", sa.String(16), nullable=False),
        sa.Column("input_state", sa.String(16), nullable=False),
        sa.Column("execution_state", sa.String(24), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("input_revision", sa.BigInteger(), nullable=False),
        sa.Column("next_input_ordinal", sa.Integer(), nullable=False),
        sa.Column("client_execution_id", postgresql.UUID(as_uuid=False), nullable=False),
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
        sa.Column("payload_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_rows", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "progress",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
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
        sa.ForeignKeyConstraint(
            ["job_id"],
            [f"{schema}.job_identities.job_id"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("operation IN ('fit', 'predict')", name="jobs_operation_ck"),
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
        sa.CheckConstraint("input_revision >= 0", name="jobs_input_revision_ck"),
        sa.CheckConstraint("next_input_ordinal >= 0", name="jobs_next_ordinal_ck"),
        sa.CheckConstraint("fencing_token >= 1", name="jobs_fencing_token_ck"),
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
        schema=schema,
    )
    op.create_index(
        "jobs_queue_idx",
        "jobs",
        ["execution_state", "selected_device", "queue_sequence"],
        schema=schema,
    )
    op.create_index(
        "jobs_owner_state_idx",
        "jobs",
        ["owner_subject", "execution_state"],
        schema=schema,
    )
    op.create_index(
        "jobs_queue_sequence_idx",
        "jobs",
        ["queue_sequence"],
        unique=True,
        schema=schema,
        postgresql_where=sa.text("queue_sequence IS NOT NULL"),
    )
    op.create_index(
        "jobs_input_waiting_idx",
        "jobs",
        ["waiting_for_input", "input_waiting_since"],
        schema=schema,
        postgresql_where=sa.text("waiting_for_input"),
    )

    op.create_table(
        "input_uploads",
        sa.Column("upload_token", sa.String(128), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("payload_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("client_execution_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("candidate_path", sa.Text(), nullable=False),
        sa.Column("storage_class", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], [f"{schema}.jobs.job_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("job_id", "ordinal", name="input_uploads_job_ordinal_uq"),
        sa.UniqueConstraint("job_id", "payload_id", name="input_uploads_job_payload_uq"),
        sa.CheckConstraint("ordinal >= 0", name="input_uploads_ordinal_ck"),
        sa.CheckConstraint("fencing_token >= 1", name="input_uploads_fencing_token_ck"),
        sa.CheckConstraint(
            "storage_class IN ('runtime', 'recovery')",
            name="input_uploads_storage_class_ck",
        ),
        schema=schema,
    )

    op.create_table(
        "job_inputs",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("payload_id", postgresql.UUID(as_uuid=False), nullable=False),
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
        sa.ForeignKeyConstraint(["job_id"], [f"{schema}.jobs.job_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id", "ordinal", name="job_inputs_pk"),
        sa.UniqueConstraint("job_id", "payload_id", name="job_inputs_job_payload_uq"),
        sa.UniqueConstraint("job_id", "commit_revision", name="job_inputs_job_revision_uq"),
        sa.UniqueConstraint("relative_path", name="job_inputs_relative_path_uq"),
        sa.CheckConstraint("ordinal >= 0", name="job_inputs_ordinal_ck"),
        sa.CheckConstraint("commit_revision > 0", name="job_inputs_revision_ck"),
        sa.CheckConstraint("rows >= 0", name="job_inputs_rows_ck"),
        sa.CheckConstraint("batches >= 0", name="job_inputs_batches_ck"),
        sa.CheckConstraint("bytes >= 0", name="job_inputs_bytes_ck"),
        sa.CheckConstraint("source_width > 0", name="job_inputs_source_width_ck"),
        sa.CheckConstraint("feature_dim > 0", name="job_inputs_feature_dim_ck"),
        sa.CheckConstraint(
            "storage_class IN ('runtime', 'recovery')",
            name="job_inputs_storage_class_ck",
        ),
        schema=schema,
    )
    op.create_index(
        "job_inputs_revision_idx",
        "job_inputs",
        ["job_id", "commit_revision"],
        schema=schema,
    )

    _create_attempt_tables(schema)
    _create_output_and_idempotency_tables(schema)

    op.create_foreign_key(
        "models_producing_job_id_fkey",
        "models",
        "jobs",
        ["producing_job_id"],
        ["job_id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )


def _create_attempt_tables(schema: str) -> None:
    op.create_table(
        "job_attempts",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("attempt_id", postgresql.UUID(as_uuid=False), nullable=False),
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
        sa.ForeignKeyConstraint(["job_id"], [f"{schema}.jobs.job_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id", "attempt", name="job_attempts_pk"),
        sa.UniqueConstraint("attempt_id", name="job_attempts_attempt_id_uq"),
        sa.CheckConstraint("attempt > 0", name="job_attempts_attempt_ck"),
        sa.CheckConstraint("selected_device IN ('cpu', 'cuda')", name="job_attempts_device_ck"),
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
    op.create_table(
        "training_recovery_checkpoints",
        sa.Column("job_id", postgresql.UUID(as_uuid=False), nullable=False),
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
            ["job_id", "attempt"],
            [f"{schema}.job_attempts.job_id", f"{schema}.job_attempts.attempt"],
            ondelete="CASCADE",
            name="training_recovery_checkpoints_attempt_fk",
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
        sa.CheckConstraint("generation > 0", name="training_recovery_checkpoints_generation_ck"),
        sa.CheckConstraint("attempt > 0", name="training_recovery_checkpoints_attempt_ck"),
        sa.CheckConstraint("bytes > 0", name="training_recovery_checkpoints_bytes_ck"),
        sa.CheckConstraint(
            "completed_epochs > 0",
            name="training_recovery_checkpoints_epochs_ck",
        ),
        sa.CheckConstraint("global_step >= 0", name="training_recovery_checkpoints_step_ck"),
        schema=schema,
    )


def _create_output_and_idempotency_tables(schema: str) -> None:
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
        "idempotency_records",
        sa.Column("owner_subject", sa.String(256), nullable=False),
        sa.Column("action_name", sa.String(128), nullable=False),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("response", postgresql.JSONB(), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=False)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"],
            [f"{schema}.job_identities.job_id"],
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
    op.create_index(
        "output_tickets_expiry_idx",
        "output_tickets",
        ["expires_at"],
        schema=schema,
    )


def downgrade() -> None:
    raise RuntimeError(
        "Flight v3 migration is destructive and cannot be downgraded; "
        "restore a pre-cutover database backup instead"
    )
