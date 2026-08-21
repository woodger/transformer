"""Replace local API tokens with Ory Hydra introspection.

Revision ID: 0014
Revises: 0013
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    op.execute(sa.text(
        "DROP TRIGGER IF EXISTS api_access_tokens_notify "
        f"ON {quoted}.api_access_tokens"
    ))
    op.execute(sa.text(
        f"DROP FUNCTION IF EXISTS {quoted}.notify_auth_token_change()"
    ))
    op.drop_table("api_access_tokens", schema=schema)


def downgrade() -> None:
    raise RuntimeError(
        "local API credentials removed by the Hydra cutover cannot be restored"
    )
