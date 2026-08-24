import importlib


def _migration():
    return importlib.import_module(
        "app.service.adapters.outbound.postgres.alembic.versions."
        "0017_expiring_api_access_tokens"
    )


def test_expiration_migration_deletes_perpetual_tokens_before_cutover():
    migration = _migration()
    statement = str(migration._delete_existing_tokens_sql("transformer"))

    assert migration.revision == "0017"
    assert migration.down_revision == "0016"
    assert statement == "DELETE FROM transformer.api_access_tokens"


def test_expiration_migration_quotes_schema_identifier():
    statement = str(_migration()._delete_existing_tokens_sql('tenant"schema'))

    assert statement == 'DELETE FROM "tenant""schema".api_access_tokens'
