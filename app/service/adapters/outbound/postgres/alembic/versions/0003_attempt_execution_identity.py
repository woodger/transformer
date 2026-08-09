"""Add equality-fenced worker attempt identity.

Revision ID: 0003
Revises: 0002
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = schema.replace('"', '""')
    op.add_column(
        "job_attempts",
        sa.Column(
            "attempt_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        schema=schema,
    )
    # Existing attempt identities are backfilled deterministically without a
    # PostgreSQL extension. New claims always use a service-generated UUID.
    op.execute(sa.text(
        f'UPDATE "{quoted}".job_attempts SET attempt_id = ('
        "substring(md5(job_id::text || ':' || attempt::text), 1, 8) || '-' || "
        "substring(md5(job_id::text || ':' || attempt::text), 9, 4) || '-' || "
        "substring(md5(job_id::text || ':' || attempt::text), 13, 4) || '-' || "
        "substring(md5(job_id::text || ':' || attempt::text), 17, 4) || '-' || "
        "substring(md5(job_id::text || ':' || attempt::text), 21, 12)"
        ")::uuid WHERE attempt_id IS NULL"
    ))
    op.alter_column(
        "job_attempts",
        "attempt_id",
        nullable=False,
        schema=schema,
    )
    op.create_unique_constraint(
        "job_attempts_attempt_id_uq",
        "job_attempts",
        ["attempt_id"],
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_constraint(
        "job_attempts_attempt_id_uq",
        "job_attempts",
        schema=schema,
        type_="unique",
    )
    op.drop_column("job_attempts", "attempt_id", schema=schema)

