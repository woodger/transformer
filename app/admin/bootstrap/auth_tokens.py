from __future__ import annotations

import secrets
import sys
from typing import Protocol

from app.admin.adapters.outbound.hydra import (
    HydraAdministrationError,
    HydraOAuthClientAdministration,
)
from app.admin.cli.auth_tokens import print_issued, print_list, print_revoked
from app.service.adapters.outbound.hydra.config import (
    load_hydra_introspection_config,
)


class AuthTokenArguments(Protocol):
    tokens_action: str
    client_id: str | None
    token_id: str


def run(args: AuthTokenArguments) -> None:
    try:
        endpoint = load_hydra_introspection_config().endpoint
        administration = HydraOAuthClientAdministration(endpoint)
        if args.tokens_action == "issue":
            client_id = (
                args.client_id
                if args.client_id is not None
                else f"trf-{secrets.token_hex(10)}"
            )
            print_issued(administration.issue(client_id))
            return
        if args.tokens_action == "list":
            print_list(administration.list())
            return
        print_revoked(administration.revoke(args.token_id))
    except (HydraAdministrationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


__all__ = ["run"]
