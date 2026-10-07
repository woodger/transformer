"""Удалить поколения, несовместимые с Semantic v6 и Flight v23.

Идентификатор ревизии: 0030
Предыдущая ревизия: 0029
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    """Очистить данные без читателя Semantic v6 и checkpoint v13."""

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
              'revision 0030 requires all jobs to be terminal';
          END IF;
        END
        $migration$
    """))

    # Revision языка входит в D1 и definition модели даже без регуляризатора.
    # Старые generations и их recovery нельзя переинтерпретировать как v6.
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
        "revision 0030 is a destructive public-contract clean cut and cannot be downgraded"
    )
