from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cli.args import parse_args


class CliArguments(Protocol):
    action: str
    flight_action: str
    tokens_action: str
    migrations_action: str
    models_action: str
    host: str | None
    port: int | None
    tls_cert_file: str | None
    tls_key_file: str | None
    tls_ca_file: str | None
    tls_require_client_cert: bool | None

def main() -> None:
    args = cast(CliArguments, parse_args())

    if args.action == "flight" and args.flight_action == "serve":
        from app.service.bootstrap.application import run_from_args

        run_from_args(args)
        return

    if args.action == "auth":
        from app.admin.bootstrap.auth_tokens import AuthTokenArguments, run

        run(cast(AuthTokenArguments, args))
        return

    if args.action == "db":
        from app.admin.bootstrap.db_migrations import MigrationArguments, run

        run(cast(MigrationArguments, args))
        return

    if args.action == "models":
        from app.admin.bootstrap.models import ModelArguments, run

        run(cast(ModelArguments, args))
        return

    raise RuntimeError(f"unsupported command: {args.action}")


if __name__ == "__main__":
    main()
