import importlib


def _migration():
    return importlib.import_module(
        "app.service.adapters.outbound.postgres.alembic.versions."
        "0016_normalize_api_access_token_storage"
    )


def test_legacy_token_storage_migration_is_forward_only_and_digest_preserving():
    migration = _migration()
    statement = " ".join(
        str(migration._legacy_conversion_sql("transformer")).split()
    )

    assert migration.revision == "0016"
    assert migration.down_revision == "0015"
    assert "column_name = 'token'" in statement
    assert "sha256(convert_to(token, 'UTF8'))" in statement
    assert "ALTER COLUMN token_digest SET NOT NULL" in statement
    assert "DROP COLUMN token" in statement
    assert "UNIQUE (token_digest)" in statement
    assert "CHECK (token_digest ~ '^[0-9a-f]{64}$')" in statement


def test_legacy_token_storage_migration_quotes_schema_as_literal_and_identifier():
    statement = str(_migration()._legacy_conversion_sql("tenant'schema"))

    assert "table_schema = 'tenant''schema'" in statement
    assert 'ALTER TABLE "tenant\'schema".api_access_tokens' in statement
