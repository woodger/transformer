import importlib


def _migration():
    return importlib.import_module(
        "app.service.adapters.outbound.postgres.alembic.versions."
        "0018_delete_revoked_api_tokens"
    )


def test_deletion_migration_removes_existing_revoked_tokens():
    migration = _migration()
    statement = str(migration._delete_revoked_tokens_sql("transformer"))

    assert migration.revision == "0018"
    assert migration.down_revision == "0017"
    assert statement == (
        "DELETE FROM transformer.api_access_tokens WHERE revoked_at IS NOT NULL"
    )


def test_deletion_migration_quotes_schema_identifier():
    statement = str(_migration()._delete_revoked_tokens_sql('tenant"schema'))

    assert statement == (
        'DELETE FROM "tenant""schema".api_access_tokens WHERE revoked_at IS NOT NULL'
    )
