"""Удалить поколения, несовместимые с Semantic v7 и Flight v24.

Идентификатор ревизии: 0031
Предыдущая ревизия: 0030
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    """Очистить данные без читателя Semantic v7 и checkpoint v14."""

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
              'revision 0031 requires all jobs to be terminal';
          END IF;
        END
        $migration$
    """))

    # Revision языка входит в D1 и definition модели даже без регуляризатора.
    # Старые generations и их recovery нельзя переинтерпретировать как v7.
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
        "revision 0031 is a destructive public-contract clean cut and cannot be downgraded"
    )
