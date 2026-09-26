"""Принять чистый переход контрактов среды выполнения без привязки к вызывающей стороне.

Идентификатор ревизии: 0024
Предыдущая ревизия: 0023
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.execute(sa.text(f"""
        DO $migration$
        BEGIN
          IF EXISTS (
            SELECT 1
            FROM "{schema}".jobs
            WHERE execution_state NOT IN ('SUCCEEDED', 'FAILED', 'CANCELLED')
          ) THEN
            RAISE EXCEPTION
              'revision 0024 requires all Flight v10 jobs to be terminal';
          END IF;
        END
        $migration$
    """))

    for table in (
        "metrics_outbox",
        "fit_run_summary_artifacts",
        "training_metrics_artifacts",
        "model_aliases",
        "models",
        "deleted_models",
        "idempotency_records",
        "jobs",
        "job_identities",
    ):
        op.execute(sa.text(f'DELETE FROM "{schema}"."{table}"'))

    op.alter_column(
        "jobs",
        "ml_contract",
        new_column_name="model_contract",
        existing_type=postgresql.JSONB(),
        existing_nullable=False,
        schema=schema,
    )
    op.add_column(
        "jobs",
        sa.Column("semantic_digests", postgresql.JSONB(), nullable=False),
        schema=schema,
    )
    op.alter_column(
        "jobs",
        "source_encoding",
        existing_type=postgresql.JSONB(),
        existing_nullable=True,
        nullable=False,
        schema=schema,
    )

    op.drop_constraint(
        "models_ml_contract_ck",
        "models",
        schema=schema,
        type_="check",
    )
    op.alter_column(
        "models",
        "ml_contract",
        new_column_name="model_contract",
        existing_type=postgresql.JSONB(),
        existing_nullable=True,
        nullable=False,
        schema=schema,
    )
    op.drop_column("models", "objective_config_sha256", schema=schema)
    op.add_column(
        "models",
        sa.Column("semantic_digests", postgresql.JSONB(), nullable=False),
        schema=schema,
    )
    op.alter_column(
        "models",
        "data_contract",
        existing_type=postgresql.JSONB(),
        existing_nullable=True,
        nullable=False,
        schema=schema,
    )
    op.alter_column(
        "models",
        "data_contract_sha256",
        existing_type=sa.String(64),
        existing_nullable=True,
        nullable=False,
        schema=schema,
    )

    op.add_column(
        "training_recovery_checkpoints",
        sa.Column("input_revision", sa.BigInteger(), nullable=False),
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    for table in (
        "metrics_outbox",
        "fit_run_summary_artifacts",
        "training_metrics_artifacts",
        "model_aliases",
        "models",
        "deleted_models",
        "idempotency_records",
        "jobs",
        "job_identities",
    ):
        op.execute(sa.text(f'DELETE FROM "{schema}"."{table}"'))

    op.alter_column(
        "jobs",
        "source_encoding",
        existing_type=postgresql.JSONB(),
        existing_nullable=False,
        nullable=True,
        schema=schema,
    )
    op.drop_column(
        "training_recovery_checkpoints",
        "input_revision",
        schema=schema,
    )

    op.alter_column(
        "models",
        "data_contract_sha256",
        existing_type=sa.String(64),
        existing_nullable=False,
        nullable=True,
        schema=schema,
    )
    op.alter_column(
        "models",
        "data_contract",
        existing_type=postgresql.JSONB(),
        existing_nullable=False,
        nullable=True,
        schema=schema,
    )
    op.drop_column("models", "semantic_digests", schema=schema)
    op.add_column(
        "models",
        sa.Column("objective_config_sha256", sa.String(64)),
        schema=schema,
    )
    op.alter_column(
        "models",
        "model_contract",
        new_column_name="ml_contract",
        existing_type=postgresql.JSONB(),
        existing_nullable=False,
        nullable=True,
        schema=schema,
    )
    op.create_check_constraint(
        "models_ml_contract_ck",
        "models",
        "(ml_contract IS NULL AND objective_config_sha256 IS NULL) OR "
        "(ml_contract IS NOT NULL AND objective_config_sha256 IS NOT NULL "
        "AND data_contract IS NOT NULL AND data_contract_sha256 IS NOT NULL)",
        schema=schema,
    )

    op.drop_column("jobs", "semantic_digests", schema=schema)
    op.alter_column(
        "jobs",
        "model_contract",
        new_column_name="ml_contract",
        existing_type=postgresql.JSONB(),
        existing_nullable=False,
        schema=schema,
    )
