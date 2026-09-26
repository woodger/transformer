"""Сохранить инициализацию модели и разрешить опубликованные модели-родители для fit.

Идентификатор ревизии: 0021
Предыдущая ревизия: 0020
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "jobs",
        sa.Column("initialization", postgresql.JSONB()),
        schema=schema,
    )
    jobs = sa.table(
        "jobs",
        sa.column("operation", sa.String()),
        sa.column("initialization", postgresql.JSONB()),
        schema=schema,
    )
    op.execute(
        jobs.update()
        .where(jobs.c.operation == "fit")
        .values(
            initialization=sa.text("'{\"kind\": \"random\"}'::jsonb")
        )
    )
    op.drop_constraint(
        "jobs_operation_fields_ck",
        "jobs",
        schema=schema,
        type_="check",
    )
    op.create_check_constraint(
        "jobs_operation_fields_ck",
        "jobs",
        "(operation = 'fit' AND model_label IS NOT NULL "
        "AND initialization IS NOT NULL) OR "
        "(operation = 'predict' AND model_label IS NULL "
        "AND resolved_model_ref IS NOT NULL AND initialization IS NULL)",
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    connection = op.get_bind()
    jobs = sa.table(
        "jobs",
        sa.column("operation", sa.String()),
        sa.column("resolved_model_ref", sa.String()),
        schema=schema,
    )
    warm_start_count = connection.scalar(
        sa.select(sa.func.count())
        .select_from(jobs)
        .where(
            jobs.c.operation == "fit",
            jobs.c.resolved_model_ref.is_not(None),
        )
    )
    if warm_start_count:
        raise RuntimeError(
            "revision 0021 cannot be downgraded while published-model fit "
            "jobs exist"
        )
    op.drop_constraint(
        "jobs_operation_fields_ck",
        "jobs",
        schema=schema,
        type_="check",
    )
    op.create_check_constraint(
        "jobs_operation_fields_ck",
        "jobs",
        "(operation = 'fit' AND model_label IS NOT NULL "
        "AND resolved_model_ref IS NULL) OR "
        "(operation = 'predict' AND model_label IS NULL "
        "AND resolved_model_ref IS NOT NULL)",
        schema=schema,
    )
    op.drop_column("jobs", "initialization", schema=schema)
