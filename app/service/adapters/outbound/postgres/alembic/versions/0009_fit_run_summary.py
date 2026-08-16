"""Add recovery-safe timing and terminal fit run summaries.

Revision ID: 0009
Revises: 0008
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted_schema = op.get_bind().dialect.identifier_preparer.quote(schema)
    op.add_column(
        "job_attempts",
        sa.Column("queue_entered_at", sa.DateTime(timezone=True)),
        schema=schema,
    )
    op.add_column(
        "job_attempts",
        sa.Column("worker_ready_at", sa.DateTime(timezone=True)),
        schema=schema,
    )
    op.add_column(
        "job_attempts",
        sa.Column("worker_completed_at", sa.DateTime(timezone=True)),
        schema=schema,
    )
    # Historical attempts cannot recover their former queue entry boundary.
    # Backfill a neutral zero wait; new attempts persist the exact boundary.
    op.execute(sa.text(
        f"UPDATE {quoted_schema}.job_attempts "
        "SET queue_entered_at = claimed_at"
    ))
    op.alter_column(
        "job_attempts",
        "queue_entered_at",
        nullable=False,
        schema=schema,
    )

    op.add_column(
        "training_metric_intervals",
        sa.Column("checkpoint_serialization_ms", sa.Float()),
        schema=schema,
    )
    op.add_column(
        "training_metric_intervals",
        sa.Column("checkpoint_publication_ms", sa.Float()),
        schema=schema,
    )
    op.create_check_constraint(
        "training_metric_intervals_checkpoint_serialization_ck",
        "training_metric_intervals",
        "checkpoint_serialization_ms >= 0",
        schema=schema,
    )
    op.create_check_constraint(
        "training_metric_intervals_checkpoint_publication_ck",
        "training_metric_intervals",
        "checkpoint_publication_ms >= 0",
        schema=schema,
    )

    op.create_table(
        "model_run_summary_artifacts",
        sa.Column(
            "model_ref",
            sa.String(128),
            sa.ForeignKey(f"{schema}.models.model_ref", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("format", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("job_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("attempt_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("application_version", sa.String(64), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "bytes > 0",
            name="model_run_summary_artifacts_bytes_ck",
        ),
        sa.CheckConstraint(
            "attempt > 0",
            name="model_run_summary_artifacts_attempt_ck",
        ),
        sa.UniqueConstraint(
            "relative_path",
            name="model_run_summary_artifacts_relative_path_uq",
        ),
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    quoted_schema = op.get_bind().dialect.identifier_preparer.quote(schema)
    used = op.get_bind().scalar(sa.text(
        f"SELECT EXISTS (SELECT 1 FROM "
        f"{quoted_schema}.model_run_summary_artifacts)"
    ))
    if used:
        raise RuntimeError(
            "fit run summary migration cannot be downgraded after use"
        )
    op.drop_table("model_run_summary_artifacts", schema=schema)
    op.drop_constraint(
        "training_metric_intervals_checkpoint_publication_ck",
        "training_metric_intervals",
        schema=schema,
        type_="check",
    )
    op.drop_constraint(
        "training_metric_intervals_checkpoint_serialization_ck",
        "training_metric_intervals",
        schema=schema,
        type_="check",
    )
    op.drop_column(
        "training_metric_intervals",
        "checkpoint_publication_ms",
        schema=schema,
    )
    op.drop_column(
        "training_metric_intervals",
        "checkpoint_serialization_ms",
        schema=schema,
    )
    op.drop_column("job_attempts", "worker_completed_at", schema=schema)
    op.drop_column("job_attempts", "worker_ready_at", schema=schema)
    op.drop_column("job_attempts", "queue_entered_at", schema=schema)
