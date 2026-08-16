from __future__ import annotations

import sys
from typing import Protocol

from app.admin.cli.models import print_deletion, print_list
from app.service.adapters.outbound.postgres.config import load_database_config
from app.service.adapters.outbound.postgres.migrations import require_current_schema
from app.service.adapters.outbound.postgres.published_models import (
    PublishedModelStore,
)
from app.service.adapters.outbound.postgres.session import Database
from app.service.application.commands.models import ModelAdministration
from app.service.domain.model import ModelDeletionBlocked


class ModelArguments(Protocol):
    models_action: str
    model_ref: str
    discard_undelivered_metrics: bool


def run(args: ModelArguments) -> None:
    config = load_database_config()
    require_current_schema(config)
    database = Database(config)
    try:
        administration = ModelAdministration(PublishedModelStore(database))
        if args.models_action == "list":
            print_list(administration.list())
            return
        print_deletion(administration.delete(
            args.model_ref,
            discard_undelivered_metrics=(
                args.discard_undelivered_metrics
            ),
        ))
    except (LookupError, ValueError, ModelDeletionBlocked) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        database.close()


__all__ = ["run"]
