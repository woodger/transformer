"""Удалить поколения, несовместимые с Semantic v5 и Flight v22.

Revision ID: 0029
Revises: 0028
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    """Очистить состояния без читателя нового порядка нормализации."""

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
              'revision 0029 requires all jobs to be terminal';
          END IF;
        END
        $migration$
    """))

    # Active runtime не читает metadata, checkpoint, recovery descriptors или
    # телеметрию поколений без обязательного encoderNormalizationOrder.
    for table in (
        "metrics_outbox",
        "fit_run_summary_artifacts",
        "training_metrics_artifacts",
        "models",
        "deleted_models",
        "idempotency_records",
        "jobs",
        "job_identities",
    ):
        op.execute(sa.text(f'DELETE FROM "{schema}"."{table}"'))


def downgrade() -> None:
    raise RuntimeError(
        "revision 0029 is a destructive public-contract clean cut and cannot be downgraded"
    )
