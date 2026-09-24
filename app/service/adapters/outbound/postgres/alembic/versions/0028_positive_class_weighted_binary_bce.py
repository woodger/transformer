"""Удалить поколения, несовместимые с Semantic v4 и Flight v17.

Revision ID: 0028
Revises: 0027
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    """Очистить данные, для которых отсутствует reader активной границы."""

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
              'revision 0028 requires all jobs to be terminal';
          END IF;
        END
        $migration$
    """))

    # Активная граница не читает metadata, checkpoints, recovery descriptors
    # или телеметрию поколений прежних ревизий. Каскады удаляют зависимые записи.
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
        "revision 0028 is a destructive public-contract clean cut and cannot be downgraded"
    )
