from __future__ import annotations

from app.database.config import load_database_config
from app.database.migrations import (
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
    print(f"Current revision: {', '.join(status.current) or 'none'}")
    print(f"Head revision: {', '.join(status.heads) or 'none'}")
    print(f"Pending migrations: {'yes' if status.pending else 'no'}")
