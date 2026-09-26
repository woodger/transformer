"""Сохранить компактные метаданные входа блоков indexedFeature.

Идентификатор ревизии: 0023
Предыдущая ревизия: 0022
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0023"
down_revision = "0022"
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
              'revision 0023 requires all pre-v10 jobs to be terminal';
          END IF;
        END
        $migration$
    """))

    op.drop_constraint(
        "jobs_input_totals_ck",
        "jobs",
        schema=schema,
        type_="check",
    )
    op.add_column(
        "jobs",
        sa.Column("source_encoding", postgresql.JSONB(), nullable=True),
        schema=schema,
    )
    op.add_column(
        "jobs",
        sa.Column(
            "total_chunks",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        schema=schema,
    )
    op.add_column(
        "jobs",
        sa.Column(
            "total_native_rows",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        schema=schema,
    )
    op.add_column(
        "jobs",
        sa.Column("range_count", sa.BigInteger(), nullable=True),
        schema=schema,
    )
    op.create_check_constraint(
        "jobs_input_totals_ck",
        "jobs",
        "payload_count >= 0 AND total_chunks >= 0 AND total_rows >= 0 "
        "AND total_bytes >= 0",
        schema=schema,
    )
    op.create_check_constraint(
        "jobs_range_count_ck",
        "jobs",
        "range_count IS NULL OR range_count >= 0",
        schema=schema,
    )

    op.add_column(
        "job_inputs",
        sa.Column(
            "chunks",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        schema=schema,
    )
    op.add_column(
        "job_inputs",
        sa.Column(
            "native_rows",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        schema=schema,
    )
    for column in (
        "first_range_ordinal",
        "first_example_offset",
        "last_range_ordinal",
        "next_example_offset",
    ):
        op.add_column(
            "job_inputs",
            sa.Column(column, sa.BigInteger(), nullable=True),
            schema=schema,
        )
    op.create_check_constraint(
        "job_inputs_chunks_ck",
        "job_inputs",
        "chunks >= 0",
        schema=schema,
    )
    for table, column in (
        ("jobs", "total_chunks"),
        ("jobs", "total_native_rows"),
        ("job_inputs", "chunks"),
        ("job_inputs", "native_rows"),
    ):
        op.alter_column(
            table,
            column,
            server_default=None,
            schema=schema,
        )


def downgrade() -> None:
    schema = _schema()
    op.execute(sa.text(f"""
        DO $migration$
        BEGIN
          IF EXISTS (
            SELECT 1
            FROM "{schema}".jobs
            WHERE source_encoding IS NOT NULL
          ) THEN
            RAISE EXCEPTION
              'revision 0023 cannot be downgraded while Flight v10 jobs exist';
          END IF;
        END
        $migration$
    """))

    op.drop_constraint(
        "job_inputs_chunks_ck",
        "job_inputs",
        schema=schema,
        type_="check",
    )
    for column in (
        "next_example_offset",
        "last_range_ordinal",
        "first_example_offset",
        "first_range_ordinal",
        "native_rows",
        "chunks",
    ):
        op.drop_column("job_inputs", column, schema=schema)

    op.drop_constraint(
        "jobs_range_count_ck",
        "jobs",
        schema=schema,
        type_="check",
    )
    op.drop_constraint(
        "jobs_input_totals_ck",
        "jobs",
        schema=schema,
        type_="check",
    )
    for column in (
        "range_count",
        "total_native_rows",
        "total_chunks",
        "source_encoding",
    ):
        op.drop_column("jobs", column, schema=schema)
    op.create_check_constraint(
        "jobs_input_totals_ck",
        "jobs",
        "payload_count >= 0 AND total_rows >= 0 AND total_bytes >= 0",
        schema=schema,
    )
