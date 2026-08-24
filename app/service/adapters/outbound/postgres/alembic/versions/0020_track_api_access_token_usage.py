"""Track the last persisted use of API access tokens.

Revision ID: 0020
Revises: 0019
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    op.add_column(
        "api_access_tokens",
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        schema=_schema(),
    )


def downgrade() -> None:
    op.drop_column(
        "api_access_tokens",
        "last_used_at",
        schema=_schema(),
    )
