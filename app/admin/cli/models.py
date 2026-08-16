from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.service.domain.records import ModelLifecycleRecord


def print_list(records: Sequence[ModelLifecycleRecord]) -> None:
    print("MODEL REF\tOWNER\tLABEL\tGENERATION\tSTATE\tMETRICS\tCREATED AT")
    for record in records:
        metrics = record.metrics_delivery_status or "-"
        created_at = datetime.fromtimestamp(record.created_at, UTC).isoformat()
        print(
            f"{record.model_ref}\t{record.owner_subject}\t{record.label}\t"
            f"{record.generation}\t{record.state.value}\t{metrics}\t{created_at}"
        )


def print_deletion(record: ModelLifecycleRecord) -> None:
    print(f"Model: {record.model_ref}")
    print(f"State: {record.state.value}")


__all__ = ["print_deletion", "print_list"]
