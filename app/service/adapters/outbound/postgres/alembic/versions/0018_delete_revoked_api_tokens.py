"""Physically delete revoked API tokens.

Revision ID: 0018
Revises: 0017
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def _delete_revoked_tokens_sql(schema: str) -> sa.TextClause:
    quoted = postgresql.dialect().identifier_preparer.quote(schema)
    return sa.text(
        f"DELETE FROM {quoted}.api_access_tokens WHERE revoked_at IS NOT NULL"
    )


def _replace_trigger(schema: str, *, include_update: bool) -> None:
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    events = "INSERT OR UPDATE OR DELETE" if include_update else "INSERT OR DELETE"
    op.execute(
        sa.text(
            "DROP TRIGGER IF EXISTS api_access_tokens_notify "
            f"ON {quoted}.api_access_tokens"
        )
    )
    op.execute(
        sa.text(
            "CREATE TRIGGER api_access_tokens_notify "
            f"AFTER {events} ON {quoted}.api_access_tokens FOR EACH STATEMENT "
            f"EXECUTE FUNCTION {quoted}.notify_auth_token_change()"
        )
    )


def upgrade() -> None:
    schema = _schema()
    op.execute(_delete_revoked_tokens_sql(schema))
    op.drop_index(
        "api_access_tokens_active_idx",
        table_name="api_access_tokens",
        schema=schema,
    )
    op.drop_column("api_access_tokens", "revoked_at", schema=schema)
    op.create_index(
        "api_access_tokens_active_idx",
        "api_access_tokens",
        ["expires_at"],
        schema=schema,
    )
    _replace_trigger(schema, include_update=False)


def downgrade() -> None:
    schema = _schema()
    op.drop_index(
        "api_access_tokens_active_idx",
        table_name="api_access_tokens",
        schema=schema,
    )
    op.add_column(
        "api_access_tokens",
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        schema=schema,
    )
    op.create_index(
        "api_access_tokens_active_idx",
        "api_access_tokens",
        ["expires_at"],
        schema=schema,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    _replace_trigger(schema, include_update=True)
