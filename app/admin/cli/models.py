from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.service.domain.records import ModelLifecycleRecord


def print_list(records: Sequence[ModelLifecycleRecord]) -> None:
    print(
        "MODEL REF\tOWNER\tLABEL\tGENERATION\tSTATE\t"
        "CREATED AT\tDELETED AT"
    )
    for model in records:
        created_at = datetime.fromtimestamp(model.created_at, UTC).isoformat()
        deleted_at = (
            ""
            if model.deleted_at is None
            else datetime.fromtimestamp(model.deleted_at, UTC).isoformat()
        )
        print(
            f"{model.model_ref}\t{model.owner_subject}\t{model.label}\t"
            f"{model.generation}\t{model.state.value}\t{created_at}\t"
            f"{deleted_at}"
        )


def print_deletion(record: ModelLifecycleRecord) -> None:
    print(f"Model: {record.model_ref}")
    print(f"State: {record.state.value}")


__all__ = ["print_deletion", "print_list"]
