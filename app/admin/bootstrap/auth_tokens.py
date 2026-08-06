from __future__ import annotations

from app.admin.cli.auth_tokens import print_issued, print_list, print_revoked
from app.service.adapters.outbound.postgres.config import load_database_config
from app.service.adapters.outbound.postgres.migrations import require_current_schema
from app.service.adapters.outbound.postgres.session import Database
from app.service.adapters.outbound.postgres.tokens import AccessTokenStore
from app.service.application.commands.access_tokens import AccessTokenAdministration


def run(args) -> None:
    config = load_database_config()
    require_current_schema(config)
    database = Database(config)
    try:
        administration = AccessTokenAdministration(AccessTokenStore(database))
        if args.tokens_action == "issue":
            print_issued(administration.issue(args.subject))
            return
        if args.tokens_action == "list":
            print_list(administration.list())
            return
        print_revoked(administration.revoke(args.token_id))
    finally:
        database.close()
