"""Add durable training recovery and device-aware attempts.

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = schema.replace('"', '""')

    # Flight v2 is a deliberate breaking cutover. Runtime job identities,
    # tickets and idempotency responses from v1 cannot be interpreted through
    # the v2 state and descriptor contract. Published models use ON DELETE
    # SET NULL provenance and remain available; aliases and access tokens are
    # independent of jobs.
    op.execute(sa.text(
        f'DELETE FROM "{quoted}".idempotency_records'
    ))
    op.execute(sa.text(f'DELETE FROM "{quoted}".jobs'))
    op.execute(sa.text(
        f'DELETE FROM "{quoted}".runtime_state '
        "WHERE key = 'storage_epoch'"
    ))

    op.drop_constraint("jobs_state_ck", "jobs", schema=schema, type_="check")
    op.create_check_constraint(
        "jobs_state_ck",
        "jobs",
        "state IN ('UPLOADING', 'SEALED', 'QUEUED', 'RUNNING', "
        "'RETRYING', 'SUCCEEDED', 'FAILED', 'CANCELLING', 'CANCELLED')",
        schema=schema,
    )

    op.add_column(
        "job_inputs",
        sa.Column(
            "storage_class",
            sa.String(16),
            nullable=False,
            server_default="runtime",
        ),
        schema=schema,
    )
    op.create_check_constraint(
        "job_inputs_storage_class_ck",
        "job_inputs",
        "storage_class IN ('runtime', 'recovery')",
        schema=schema,
    )
    op.add_column(
        "input_uploads",
        sa.Column(
            "storage_class",
            sa.String(16),
            nullable=False,
            server_default="runtime",
        ),
        schema=schema,
    )
    op.create_check_constraint(
        "input_uploads_storage_class_ck",
        "input_uploads",
        "storage_class IN ('runtime', 'recovery')",
        schema=schema,
    )

    op.add_column(
        "job_attempts",
        sa.Column("device_id", sa.String(128)),
        schema=schema,
    )
    op.add_column(
        "job_attempts",
        sa.Column("resume_generation", sa.Integer()),
        schema=schema,
    )
    op.create_check_constraint(
        "job_attempts_resume_generation_ck",
        "job_attempts",
        "resume_generation IS NULL OR resume_generation > 0",
        schema=schema,
    )
    op.create_check_constraint(
        "job_attempts_device_assignment_ck",
        "job_attempts",
        "(selected_device = 'cpu' AND device_id IS NULL) OR "
        "(selected_device = 'cuda' AND device_id IS NOT NULL)",
        schema=schema,
    )

    op.create_table(
        "training_recovery_checkpoints",
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
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["job_id", "attempt"],
            [
                f"{schema}.job_attempts.job_id",
                f"{schema}.job_attempts.attempt",
            ],
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


def downgrade() -> None:
    schema = _schema()
    quoted = schema.replace('"', '""')

    # The v2 job and idempotency records cannot be represented by the v1
    # contract. Published models and access tokens remain independent.
    op.execute(sa.text(
        f'DELETE FROM "{quoted}".idempotency_records'
    ))
    op.execute(sa.text(f'DELETE FROM "{quoted}".jobs'))
    op.execute(sa.text(
        f'DELETE FROM "{quoted}".runtime_state '
        "WHERE key = 'storage_epoch'"
    ))
    op.drop_table("training_recovery_checkpoints", schema=schema)
    op.drop_constraint(
        "job_attempts_device_assignment_ck",
        "job_attempts",
        schema=schema,
        type_="check",
    )
    op.drop_constraint(
        "job_attempts_resume_generation_ck",
        "job_attempts",
        schema=schema,
        type_="check",
    )
    op.drop_column("job_attempts", "resume_generation", schema=schema)
    op.drop_column("job_attempts", "device_id", schema=schema)
    op.drop_constraint(
        "input_uploads_storage_class_ck",
        "input_uploads",
        schema=schema,
        type_="check",
    )
    op.drop_column("input_uploads", "storage_class", schema=schema)
    op.drop_constraint(
        "job_inputs_storage_class_ck",
        "job_inputs",
        schema=schema,
        type_="check",
    )
    op.drop_column("job_inputs", "storage_class", schema=schema)
    op.drop_constraint("jobs_state_ck", "jobs", schema=schema, type_="check")
    op.create_check_constraint(
        "jobs_state_ck",
        "jobs",
        "state IN ('UPLOADING', 'SEALED', 'QUEUED', 'RUNNING', "
        "'SUCCEEDED', 'FAILED', 'CANCELLING', 'CANCELLED')",
        schema=schema,
    )
