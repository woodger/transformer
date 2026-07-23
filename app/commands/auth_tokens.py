from __future__ import annotations

from app.database import Database, load_database_config
from app.database.migrations import require_current_schema
from app.database.tokens import AccessTokenStore


def run(args) -> None:
    config = load_database_config()
    require_current_schema(config)
    database = Database(config)
    try:
        store = AccessTokenStore(database)
        if args.tokens_action == "issue":
            record = store.issue(args.subject)
            print(f"Token ID: {record.token_id}")
            print(f"Subject: {record.subject}")
            print(f"Token: {record.token}")
            return
        if args.tokens_action == "list":
            print("TOKEN ID\tSUBJECT\tCREATED AT\tSTATUS")
            for record in store.list():
                status = "revoked" if record.revoked_at is not None else "active"
                print(
                    f"{record.token_id}\t{record.subject}\t"
                    f"{record.created_at.isoformat()}\t{status}"
                )
            return
        record = store.revoke(args.token_id)
        print(f"Revoked token: {record.token_id}")
    finally:
        database.close()
