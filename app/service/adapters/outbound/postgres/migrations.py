from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect

from app.project import PROJECT_ROOT
from app.service.adapters.outbound.postgres.config import DatabaseConfig
from app.service.adapters.outbound.postgres.session import Database


@dataclass(frozen=True)
class MigrationStatus:
    current: tuple[str, ...]
    heads: tuple[str, ...]

    @property
    def pending(self) -> bool:
        return set(self.current) != set(self.heads)


def alembic_config(database_config: DatabaseConfig) -> Config:
    config = Config(toml_file=str(Path(PROJECT_ROOT) / "pyproject.toml"))
    config.set_main_option(
        "script_location",
        str(
            Path(PROJECT_ROOT)
            / "app"
            / "service"
            / "adapters"
            / "outbound"
            / "postgres"
            / "alembic"
        ),
    )
    config.attributes["database_config"] = database_config
    config.attributes["schema"] = database_config.schema
    return config


def migration_status(database_config: DatabaseConfig) -> MigrationStatus:
    database = Database(database_config)
    try:
        script = ScriptDirectory.from_config(alembic_config(database_config))
        heads = tuple(script.get_heads())
        with database.engine.connect() as connection:
            if not inspect(connection).has_schema(database_config.schema):
                current = ()
            else:
                context = MigrationContext.configure(
                    connection,
                    opts={"version_table_schema": database_config.schema},
                )
                current = tuple(context.get_current_heads())
        return MigrationStatus(current=current, heads=heads)
    finally:
        database.close()


def apply_migrations(database_config: DatabaseConfig) -> MigrationStatus:
    command.upgrade(alembic_config(database_config), "head")
    return migration_status(database_config)


def rollback_migration(database_config: DatabaseConfig) -> MigrationStatus:
    status = migration_status(database_config)
    if not status.current:
        raise RuntimeError("database schema has no applied migration to roll back")
    command.downgrade(alembic_config(database_config), "-1")
    return migration_status(database_config)


def require_current_schema(database_config: DatabaseConfig) -> None:
    status = migration_status(database_config)
    if status.pending:
        current = ", ".join(status.current) or "none"
        heads = ", ".join(status.heads) or "none"
        raise RuntimeError(
            "PostgreSQL schema is not current "
            f"(current: {current}; expected: {heads}); "
            "run 'transformer db migrations apply'"
        )
