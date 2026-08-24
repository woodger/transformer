import importlib


def _migration():
    return importlib.import_module(
        "app.service.adapters.outbound.postgres.alembic.versions."
        "0020_track_api_access_token_usage"
    )


def test_usage_migration_adds_nullable_timezone_timestamp(monkeypatch):
    migration = _migration()
    calls = []

    monkeypatch.setattr(migration, "_schema", lambda: "tenant")
    monkeypatch.setattr(
        migration.op,
        "add_column",
        lambda table, column, *, schema: calls.append(
            (table, column, schema)
        ),
    )

    migration.upgrade()

    assert migration.revision == "0020"
    assert migration.down_revision == "0019"
    assert len(calls) == 1
    table, column, schema = calls[0]
    assert table == "api_access_tokens"
    assert column.name == "last_used_at"
    assert column.nullable is True
    assert column.type.timezone is True
    assert schema == "tenant"


def test_usage_migration_downgrade_drops_usage_timestamp(monkeypatch):
    migration = _migration()
    calls = []

    monkeypatch.setattr(migration, "_schema", lambda: "tenant")
    monkeypatch.setattr(
        migration.op,
        "drop_column",
        lambda table, column, *, schema: calls.append(
            (table, column, schema)
        ),
    )

    migration.downgrade()

    assert calls == [("api_access_tokens", "last_used_at", "tenant")]
