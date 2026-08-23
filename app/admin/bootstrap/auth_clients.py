from __future__ import annotations

import sys
from typing import Protocol

from app.admin.adapters.outbound.hydra import (
    HydraAdministrationError,
    HydraOAuthClientAdministration,
)
from app.admin.cli.auth_clients import print_created, print_deleted, print_list
from app.service.adapters.outbound.hydra.config import (
    load_hydra_introspection_config,
)


class AuthClientArguments(Protocol):
    clients_action: str
    client_id: str
    name: str | None


def run(args: AuthClientArguments) -> None:
    try:
        endpoint = load_hydra_introspection_config().endpoint
        administration = HydraOAuthClientAdministration(endpoint)
        if args.clients_action == "create":
            client_name = args.client_id if args.name is None else args.name
            print_created(administration.create(args.client_id, client_name))
            return
        if args.clients_action == "list":
            print_list(administration.list())
            return
        print_deleted(administration.delete(args.client_id))
    except (HydraAdministrationError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


__all__ = ["run"]
