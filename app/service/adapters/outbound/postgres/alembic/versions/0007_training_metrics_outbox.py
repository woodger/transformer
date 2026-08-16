"""Add durable training metrics artifacts and OpenSearch outbox.

Revision ID: 0007
Revises: 0006
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.create_table(
        "training_metric_intervals",
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "generation > 0",
            name="training_metric_intervals_generation_ck",
        ),
        sa.CheckConstraint(
            "attempt > 0",
            name="training_metric_intervals_attempt_ck",
        ),
        sa.ForeignKeyConstraint(
            ("job_id", "attempt"),
            (
                f"{schema}.job_attempts.job_id",
                f"{schema}.job_attempts.attempt",
            ),
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
        schema=schema,
    )
    op.create_table(
        "model_metrics_artifacts",
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("format", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("application_version", sa.String(64), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "bytes > 0",
            name="model_metrics_artifacts_bytes_ck",
        ),
        sa.CheckConstraint(
            "row_count > 0",
            name="model_metrics_artifacts_rows_ck",
        ),
        sa.CheckConstraint(
            "attempt > 0",
            name="model_metrics_artifacts_attempt_ck",
        ),
        sa.ForeignKeyConstraint(
            ("model_ref",),
            (f"{schema}.models.model_ref",),
            name="model_metrics_artifacts_model_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "model_ref",
            name="model_metrics_artifacts_pk",
        ),
        sa.UniqueConstraint(
            "relative_path",
            name="model_metrics_artifacts_relative_path_uq",
        ),
        schema=schema,
    )
    op.create_table(
        "metrics_outbox",
        sa.Column("model_ref", sa.String(128), nullable=False),
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
        sa.CheckConstraint(
            "status IN ('PENDING', 'BLOCKED', 'DELIVERED')",
            name="metrics_outbox_status_ck",
        ),
        sa.CheckConstraint("cursor >= 0", name="metrics_outbox_cursor_ck"),
        sa.CheckConstraint(
            "attempts >= 0",
            name="metrics_outbox_attempts_ck",
        ),
        sa.ForeignKeyConstraint(
            ("model_ref",),
            (f"{schema}.model_metrics_artifacts.model_ref",),
            name="metrics_outbox_artifact_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("model_ref", name="metrics_outbox_pk"),
        schema=schema,
    )
    op.create_index(
        "metrics_outbox_pending_idx",
        "metrics_outbox",
        ["status", "next_attempt_at", "created_at"],
        unique=False,
        schema=schema,
        postgresql_where=sa.text("status = 'PENDING'"),
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_index(
        "metrics_outbox_pending_idx",
        table_name="metrics_outbox",
        schema=schema,
    )
    op.drop_table("metrics_outbox", schema=schema)
    op.drop_table("model_metrics_artifacts", schema=schema)
    op.drop_table("training_metric_intervals", schema=schema)
