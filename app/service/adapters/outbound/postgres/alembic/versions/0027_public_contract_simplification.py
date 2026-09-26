"""Удалить поколения, несовместимые с упрощённой публичной границей.

Идентификатор ревизии: 0027
Предыдущая ревизия: 0026
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    """Удалить состояние, чей публичный контракт не читается Flight v15."""

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
              'revision 0027 requires all jobs to be terminal';
          END IF;
        END
        $migration$
    """))

    # Чистый переход v15 не содержит читателя для записей v14 о заданиях,
    # моделях, восстановлении или телеметрии. Каскады внешних ключей удаляют
    # связанные квитанции, попытки, выходы и строки восстановления при удалении.
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

    op.execute(sa.text(f"""
        DELETE FROM "{schema}"."runtime_state"
        WHERE key = 'model_catalog_cursor_hmac_v1'
    """))
    op.execute(sa.text(f'DROP TABLE "{schema}"."model_aliases"'))


def downgrade() -> None:
    raise RuntimeError(
        "revision 0027 is a destructive public-contract clean cut and cannot be downgraded"
    )
