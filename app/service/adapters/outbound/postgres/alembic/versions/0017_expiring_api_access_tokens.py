"""Replace perpetual API tokens with three-month credentials.

Revision ID: 0017
Revises: 0016
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def _delete_existing_tokens_sql(schema: str) -> sa.TextClause:
    quoted = postgresql.dialect().identifier_preparer.quote(schema)
    return sa.text(f"DELETE FROM {quoted}.api_access_tokens")


def upgrade() -> None:
    schema = _schema()
    op.execute(_delete_existing_tokens_sql(schema))
    op.add_column(
        "api_access_tokens",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        schema=schema,
    )
    op.create_check_constraint(
        "api_access_tokens_expiry_order_ck",
        "api_access_tokens",
        "expires_at > created_at",
        schema=schema,
    )
    op.drop_index(
        "api_access_tokens_active_idx",
        table_name="api_access_tokens",
        schema=schema,
    )
    op.create_index(
        "api_access_tokens_active_idx",
        "api_access_tokens",
        ["expires_at"],
        schema=schema,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )


def downgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    used = op.get_bind().scalar(
        sa.text(f"SELECT EXISTS (SELECT 1 FROM {quoted}.api_access_tokens)")
    )
    if used:
        raise RuntimeError(
            "expiring API token migration cannot be downgraded after use"
        )
    op.drop_index(
        "api_access_tokens_active_idx",
        table_name="api_access_tokens",
        schema=schema,
    )
    op.create_index(
        "api_access_tokens_active_idx",
        "api_access_tokens",
        ["revoked_at"],
        schema=schema,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.drop_constraint(
        "api_access_tokens_expiry_order_ck",
        "api_access_tokens",
        schema=schema,
        type_="check",
    )
    op.drop_column("api_access_tokens", "expires_at", schema=schema)
