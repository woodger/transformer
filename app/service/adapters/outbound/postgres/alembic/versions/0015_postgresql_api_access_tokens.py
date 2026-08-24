"""Restore PostgreSQL-backed API access tokens without stored credentials.

Revision ID: 0015
Revises: 0014
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def upgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    op.create_table(
        "api_access_tokens",
        sa.Column(
            "token_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("token_digest", sa.String(64), nullable=False),
        sa.Column("subject", sa.String(256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.PrimaryKeyConstraint("token_id", name="api_access_tokens_pk"),
        sa.UniqueConstraint(
            "token_digest",
            name="api_access_tokens_token_digest_uq",
        ),
        sa.CheckConstraint(
            "token_digest ~ '^[0-9a-f]{64}$'",
            name="api_access_tokens_digest_format_ck",
        ),
        schema=schema,
    )
    op.create_index(
        "api_access_tokens_active_idx",
        "api_access_tokens",
        ["revoked_at"],
        schema=schema,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.execute(sa.text(
        f"CREATE FUNCTION {quoted}.notify_auth_token_change() RETURNS trigger "
        "LANGUAGE plpgsql AS $$ BEGIN "
        "PERFORM pg_notify('transformer_auth_tokens', 'changed'); "
        "RETURN COALESCE(NEW, OLD); END; $$"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER api_access_tokens_notify "
        "AFTER INSERT OR UPDATE OR DELETE "
        f"ON {quoted}.api_access_tokens FOR EACH STATEMENT "
        f"EXECUTE FUNCTION {quoted}.notify_auth_token_change()"
    ))


def downgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    used = op.get_bind().scalar(sa.text(
        f"SELECT EXISTS (SELECT 1 FROM {quoted}.api_access_tokens)"
    ))
    if used:
        raise RuntimeError(
            "API access token migration cannot be downgraded after use"
        )
    op.execute(sa.text(
        "DROP TRIGGER IF EXISTS api_access_tokens_notify "
        f"ON {quoted}.api_access_tokens"
    ))
    op.execute(sa.text(
        f"DROP FUNCTION IF EXISTS {quoted}.notify_auth_token_change()"
    ))
    op.drop_table("api_access_tokens", schema=schema)
