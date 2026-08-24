"""Remove API access token LISTEN/NOTIFY objects.

Revision ID: 0019
Revises: 0018
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def _drop_notification_sql(schema: str) -> tuple[sa.TextClause, ...]:
    quoted = postgresql.dialect().identifier_preparer.quote(schema)
    return (
        sa.text(
            "DROP TRIGGER IF EXISTS api_access_tokens_notify "
            f"ON {quoted}.api_access_tokens"
        ),
        sa.text(
            "DROP FUNCTION IF EXISTS "
            f"{quoted}.notify_auth_token_change()"
        ),
    )


def _restore_notification_sql(schema: str) -> tuple[sa.TextClause, ...]:
    quoted = postgresql.dialect().identifier_preparer.quote(schema)
    return (
        sa.text(
            f"CREATE OR REPLACE FUNCTION {quoted}.notify_auth_token_change() "
            "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
            "PERFORM pg_notify('transformer_auth_tokens', 'changed'); "
            "RETURN COALESCE(NEW, OLD); END; $$"
        ),
        sa.text(
            "CREATE TRIGGER api_access_tokens_notify "
            "AFTER INSERT OR DELETE "
            f"ON {quoted}.api_access_tokens FOR EACH STATEMENT "
            f"EXECUTE FUNCTION {quoted}.notify_auth_token_change()"
        ),
    )


def upgrade() -> None:
    for statement in _drop_notification_sql(_schema()):
        op.execute(statement)


def downgrade() -> None:
    for statement in _restore_notification_sql(_schema()):
        op.execute(statement)
