"""Cut over durable jobs and model metadata to target-aligned Flight v4.

Revision ID: 0006
Revises: 0005
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = schema.replace('"', '""')

    # Flight v3 jobs and recovery checkpoints cannot be interpreted under the
    # target-aligned objective. Access tokens and published model identities
    # remain; old models deliberately receive no v4 ML contract metadata.
    op.execute(sa.text(f'UPDATE "{quoted}".models SET producing_job_id = NULL'))
    op.execute(sa.text(f'DELETE FROM "{quoted}".idempotency_records'))
    op.execute(sa.text(f'DELETE FROM "{quoted}".jobs'))
    op.execute(sa.text(f'DELETE FROM "{quoted}".job_identities'))
    op.execute(sa.text(
        f'DELETE FROM "{quoted}".runtime_state WHERE key = \'storage_epoch\''
    ))
    op.execute(sa.text(
        f'ALTER SEQUENCE "{quoted}".job_queue_sequence_seq RESTART WITH 1'
    ))

    op.add_column(
        "jobs",
        sa.Column("ml_contract", postgresql.JSONB(), nullable=False),
        schema=schema,
    )

    op.drop_constraint(
        "models_v3_certification_ck",
        "models",
        schema=schema,
        type_="check",
    )
    op.drop_column("models", "certified_for_v3", schema=schema)
    op.add_column(
        "models",
        sa.Column("ml_contract", postgresql.JSONB()),
        schema=schema,
    )
    op.add_column(
        "models",
        sa.Column("objective_config_sha256", sa.String(64)),
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


def downgrade() -> None:
    raise RuntimeError(
        "Flight v4 migration is destructive and cannot be downgraded; "
        "restore a pre-cutover database backup instead"
    )
