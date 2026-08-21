"""Purge deleted model generations instead of retaining tombstones.

Revision ID: 0012
Revises: 0011
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)

    op.execute(sa.text(
        f"DELETE FROM {quoted}.models WHERE lifecycle_state = 'DELETED'"
    ))
    op.drop_constraint(
        "models_lifecycle_ck",
        "models",
        schema=schema,
        type_="check",
    )
    op.drop_column("models", "deleted_at", schema=schema)
    op.create_check_constraint(
        "models_lifecycle_ck",
        "models",
        "(lifecycle_state = 'AVAILABLE' "
        "AND deletion_requested_at IS NULL) OR "
        "(lifecycle_state = 'DELETING' "
        "AND deletion_requested_at IS NOT NULL)",
        schema=schema,
    )


def downgrade() -> None:
    raise RuntimeError(
        "hard-deleted model tombstones cannot be restored"
    )
