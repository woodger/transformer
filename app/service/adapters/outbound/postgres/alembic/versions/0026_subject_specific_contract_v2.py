"""Adopt the clean-cut subject-specific contract vocabulary.

Revision ID: 0026
Revises: 0025
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    """Discard artifacts whose v1 canonical bytes cannot be reinterpreted."""

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
              'revision 0026 requires all Flight v13 jobs to be terminal';
          END IF;
        END
        $migration$
    """))

    # У Semantic v1 и v2 намеренно различаются предобразы D1. Сверка при старте
    # приложения удаляет файловые артефакты после того, как записи реестра и
    # жизненного цикла перестают на них ссылаться.
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


def downgrade() -> None:
    raise RuntimeError(
        "revision 0026 is a destructive contract clean cut and cannot be downgraded"
    )
