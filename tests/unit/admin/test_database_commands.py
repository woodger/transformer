from types import SimpleNamespace

import pytest

import app.admin.bootstrap.db_migrations as migrations_command
from app.service.adapters.outbound.postgres.migrations import MigrationStatus


@pytest.mark.parametrize(
    ("action", "selected_name"),
    (
        ("status", "migration_status"),
        ("apply", "apply_migrations"),
        ("rollback", "rollback_migration"),
    ),
)
def test_database_migration_command_dispatches_action_and_prints_status(
    monkeypatch,
    capsys,
    action,
    selected_name,
):
    config = object()
    calls = []
    status = MigrationStatus(current=("0020",), heads=("0020",))
    monkeypatch.setattr(
        migrations_command,
        "load_database_config",
        lambda: config,
    )

    def selected(received):
        calls.append(received)
        return status

    for name in ("migration_status", "apply_migrations", "rollback_migration"):
        replacement = selected if name == selected_name else pytest.fail
        monkeypatch.setattr(migrations_command, name, replacement)

    migrations_command.run(
        SimpleNamespace(migrations_action=action)
    )

    assert calls == [config]
    assert capsys.readouterr().out == (
        "Current revision: 0020\n"
        "Head revision: 0020\n"
        "Pending migrations: no\n"
    )


def test_database_migration_status_prints_empty_current_revision(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        migrations_command,
        "load_database_config",
        lambda: object(),
    )
    monkeypatch.setattr(
        migrations_command,
        "migration_status",
        lambda _: MigrationStatus(current=(), heads=("0020",)),
    )

    migrations_command.run(
        SimpleNamespace(migrations_action="status")
    )

    assert capsys.readouterr().out == (
        "Current revision: none\n"
        "Head revision: 0020\n"
        "Pending migrations: yes\n"
    )
