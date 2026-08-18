"""Move durable telemetry ownership from models to fit runs.

Revision ID: 0010
Revises: 0009
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    op.create_table(
        "training_metrics_artifacts",
        sa.Column("job_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("format", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("attempt_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("application_version", sa.String(64), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
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
        schema=schema,
    )
    op.create_table(
        "fit_run_summary_artifacts",
        sa.Column("job_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("format", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("attempt_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("application_version", sa.String(64), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "bytes > 0",
            name="fit_run_summary_artifacts_bytes_ck",
        ),
        sa.CheckConstraint(
            "attempt > 0",
            name="fit_run_summary_artifacts_attempt_ck",
        ),
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
        schema=schema,
    )
    op.create_table(
        "metrics_outbox_run",
        sa.Column("job_id", sa.Uuid(as_uuid=False), nullable=False),
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
            "status IN ('PENDING', 'BLOCKED', 'DELIVERED', 'CANCELLED')",
            name="metrics_outbox_run_status_ck",
        ),
        sa.CheckConstraint(
            "cursor >= 0",
            name="metrics_outbox_run_cursor_ck",
        ),
        sa.CheckConstraint(
            "attempts >= 0",
            name="metrics_outbox_run_attempts_ck",
        ),
        sa.ForeignKeyConstraint(
            ("job_id",),
            (f"{schema}.training_metrics_artifacts.job_id",),
            name="metrics_outbox_run_artifact_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", name="metrics_outbox_run_pk"),
        schema=schema,
    )
    op.drop_index(
        "metrics_outbox_pending_idx",
        table_name="metrics_outbox",
        schema=schema,
    )
    op.drop_table("metrics_outbox", schema=schema)
    op.drop_table("model_run_summary_artifacts", schema=schema)
    op.drop_table("model_metrics_artifacts", schema=schema)
    op.rename_table(
        "metrics_outbox_run",
        "metrics_outbox",
        schema=schema,
    )
    for old, new in (
        ("metrics_outbox_run_status_ck", "metrics_outbox_status_ck"),
        ("metrics_outbox_run_cursor_ck", "metrics_outbox_cursor_ck"),
        ("metrics_outbox_run_attempts_ck", "metrics_outbox_attempts_ck"),
        ("metrics_outbox_run_artifact_fk", "metrics_outbox_artifact_fk"),
        ("metrics_outbox_run_pk", "metrics_outbox_pk"),
    ):
        op.execute(sa.text(
            f"ALTER TABLE {quoted}.metrics_outbox "
            f"RENAME CONSTRAINT {old} TO {new}"
        ))
    op.create_index(
        "metrics_outbox_pending_idx",
        "metrics_outbox",
        ["status", "next_attempt_at", "created_at"],
        schema=schema,
        postgresql_where=sa.text("status = 'PENDING'"),
    )


def downgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    used = op.get_bind().scalar(sa.text(
        f"SELECT EXISTS (SELECT 1 FROM {quoted}.training_metrics_artifacts "
        f"UNION ALL SELECT 1 FROM {quoted}.fit_run_summary_artifacts)"
    ))
    if used:
        raise RuntimeError(
            "run-owned telemetry migration cannot be downgraded after use"
        )

    op.drop_index(
        "metrics_outbox_pending_idx",
        table_name="metrics_outbox",
        schema=schema,
    )
    op.drop_table("metrics_outbox", schema=schema)
    op.drop_table("fit_run_summary_artifacts", schema=schema)
    op.drop_table("training_metrics_artifacts", schema=schema)
    _create_model_owned_tables(schema)


def _create_model_owned_tables(schema: str) -> None:
    op.create_table(
        "model_metrics_artifacts",
        sa.Column("model_ref", sa.String(128), nullable=False),
        sa.Column("format", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("attempt_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("application_version", sa.String(64), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "bytes > 0", name="model_metrics_artifacts_bytes_ck"
        ),
        sa.CheckConstraint(
            "row_count > 0", name="model_metrics_artifacts_rows_ck"
        ),
        sa.CheckConstraint(
            "attempt > 0", name="model_metrics_artifacts_attempt_ck"
        ),
        sa.ForeignKeyConstraint(
            ("model_ref",),
            (f"{schema}.models.model_ref",),
            name="model_metrics_artifacts_model_fk",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "model_ref", name="model_metrics_artifacts_pk"
        ),
        sa.UniqueConstraint(
            "relative_path",
            name="model_metrics_artifacts_relative_path_uq",
        ),
        schema=schema,
    )
    op.create_table(
        "model_run_summary_artifacts",
        sa.Column("model_ref", sa.String(128), nullable=False),
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
            "bytes > 0", name="model_run_summary_artifacts_bytes_ck"
        ),
        sa.CheckConstraint(
            "attempt > 0", name="model_run_summary_artifacts_attempt_ck"
        ),
        sa.ForeignKeyConstraint(
            ("model_ref",),
            (f"{schema}.models.model_ref",),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("model_ref"),
        sa.UniqueConstraint(
            "relative_path",
            name="model_run_summary_artifacts_relative_path_uq",
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
            "status IN ('PENDING', 'BLOCKED', 'DELIVERED', 'CANCELLED')",
            name="metrics_outbox_status_ck",
        ),
        sa.CheckConstraint("cursor >= 0", name="metrics_outbox_cursor_ck"),
        sa.CheckConstraint(
            "attempts >= 0", name="metrics_outbox_attempts_ck"
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
        schema=schema,
        postgresql_where=sa.text("status = 'PENDING'"),
    )
