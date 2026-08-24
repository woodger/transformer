import importlib


def _migration():
    return importlib.import_module(
        "app.service.adapters.outbound.postgres.alembic.versions."
        "0019_remove_api_access_token_notifications"
    )


def test_notification_migration_removes_trigger_and_function():
    migration = _migration()
    statements = tuple(
        str(statement)
        for statement in migration._drop_notification_sql("transformer")
    )

    assert migration.revision == "0019"
    assert migration.down_revision == "0018"
    assert statements == (
        "DROP TRIGGER IF EXISTS api_access_tokens_notify "
        "ON transformer.api_access_tokens",
        "DROP FUNCTION IF EXISTS transformer.notify_auth_token_change()",
    )


def test_notification_migration_quotes_schema_identifier():
    statements = tuple(
        str(statement)
        for statement in _migration()._drop_notification_sql('tenant"schema')
    )

    assert statements == (
        "DROP TRIGGER IF EXISTS api_access_tokens_notify "
        'ON "tenant""schema".api_access_tokens',
        'DROP FUNCTION IF EXISTS "tenant""schema".'
        "notify_auth_token_change()",
    )


def test_notification_migration_restores_revision_0018_objects():
    statements = " ".join(
        str(statement)
        for statement in _migration()._restore_notification_sql("transformer")
    )

    assert "pg_notify('transformer_auth_tokens', 'changed')" in statements
    assert "AFTER INSERT OR DELETE" in statements
