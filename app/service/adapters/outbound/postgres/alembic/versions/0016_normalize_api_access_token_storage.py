"""Normalize legacy API token tables to digest-only storage.

Revision ID: 0016
Revises: 0015
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def _schema() -> str:
    return context.config.attributes.get("schema", "transformer")


def _legacy_conversion_sql(schema: str) -> sa.TextClause:
    quoted = postgresql.dialect().identifier_preparer.quote(schema)
    literal = schema.replace("'", "''")
    return sa.text(f"""
DO $migration$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = '{literal}'
          AND table_name = 'api_access_tokens'
          AND column_name = 'token'
    ) THEN
        ALTER TABLE {quoted}.api_access_tokens
            ADD COLUMN IF NOT EXISTS token_digest varchar(64);

        UPDATE {quoted}.api_access_tokens
        SET token_digest = encode(
            sha256(convert_to(token, 'UTF8')),
            'hex'
        );

        ALTER TABLE {quoted}.api_access_tokens
            ALTER COLUMN token_digest SET NOT NULL,
            DROP CONSTRAINT IF EXISTS api_access_tokens_token_uq,
            DROP CONSTRAINT IF EXISTS api_access_tokens_format_ck,
            DROP CONSTRAINT IF EXISTS api_access_tokens_token_digest_uq,
            DROP CONSTRAINT IF EXISTS api_access_tokens_digest_format_ck,
            DROP COLUMN token;

        ALTER TABLE {quoted}.api_access_tokens
            ADD CONSTRAINT api_access_tokens_token_digest_uq
                UNIQUE (token_digest),
            ADD CONSTRAINT api_access_tokens_digest_format_ck
                CHECK (token_digest ~ '^[0-9a-f]{{64}}$');
    END IF;
END
$migration$
""")


def upgrade() -> None:
    schema = _schema()
    quoted = op.get_bind().dialect.identifier_preparer.quote(schema)
    op.execute(_legacy_conversion_sql(schema))
    op.execute(sa.text(
        f"DROP INDEX IF EXISTS {quoted}.api_access_tokens_active_idx"
    ))
    op.execute(sa.text(
        "CREATE INDEX api_access_tokens_active_idx "
        f"ON {quoted}.api_access_tokens (revoked_at) "
        "WHERE revoked_at IS NULL"
    ))
    op.execute(sa.text(
        f"CREATE OR REPLACE FUNCTION {quoted}.notify_auth_token_change() "
        "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
        "PERFORM pg_notify('transformer_auth_tokens', 'changed'); "
        "RETURN COALESCE(NEW, OLD); END; $$"
    ))
    op.execute(sa.text(
        "DROP TRIGGER IF EXISTS api_access_tokens_notify "
        f"ON {quoted}.api_access_tokens"
    ))
    op.execute(sa.text(
        "CREATE TRIGGER api_access_tokens_notify "
        "AFTER INSERT OR UPDATE OR DELETE "
        f"ON {quoted}.api_access_tokens FOR EACH STATEMENT "
        f"EXECUTE FUNCTION {quoted}.notify_auth_token_change()"
    ))


def downgrade() -> None:
    # Revision 0015 already defines digest-only storage. Normalization only
    # repairs databases whose physical table drifted from that revision.
    pass
