from __future__ import annotations

from app.admin.cli.db_migrations import print_status
from app.service.adapters.outbound.postgres.config import load_database_config
from app.service.adapters.outbound.postgres.migrations import (
    apply_migrations,
    migration_status,
    rollback_migration,
)


def run(args) -> None:
    config = load_database_config()
    if args.migrations_action == "status":
        status = migration_status(config)
    elif args.migrations_action == "apply":
        status = apply_migrations(config)
    else:
        status = rollback_migration(config)
    print_status(status)
