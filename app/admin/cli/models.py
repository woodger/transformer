from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.service.application.commands.models import ModelAdministrationRecord
from app.service.domain.records import ModelLifecycleRecord


def print_list(records: Sequence[ModelAdministrationRecord]) -> None:
    print("MODEL REF\tOWNER\tLABEL\tGENERATION\tSTATE\tMETRICS\tCREATED AT")
    for record in records:
        model = record.model
        metrics = record.metrics_delivery_status or "-"
        created_at = datetime.fromtimestamp(model.created_at, UTC).isoformat()
        print(
            f"{model.model_ref}\t{model.owner_subject}\t{model.label}\t"
            f"{model.generation}\t{model.state.value}\t{metrics}\t{created_at}"
        )


def print_deletion(record: ModelLifecycleRecord) -> None:
    print(f"Model: {record.model_ref}")
    print(f"State: {record.state.value}")


__all__ = ["print_deletion", "print_list"]
