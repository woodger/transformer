from __future__ import annotations

from app.service.domain.job import ExecutionState
from app.service.domain.records import OutputRecord


class PostgresOutputAccessStore:
    def __init__(self, ledger):
        self.ledger = ledger

    def execution_state(
        self,
        job_id: str,
        *,
        owner_subject: str,
    ) -> ExecutionState | None:
        job = self.ledger.get_job(job_id, owner_subject=owner_subject)
        return (
            None
            if job is None
            else ExecutionState(job["execution_state"])
        )

    def find_output(
        self,
        job_id: str,
        ordinal: int,
    ) -> OutputRecord | None:
        value = next(
            (
                item
                for item in self.ledger.list_outputs(job_id)
                if item["ordinal"] == ordinal
            ),
            None,
        )
        return None if value is None else _output_record(value)

    def issue_ticket(self, **kwargs) -> tuple[bytes, float]:
        return self.ledger.issue_ticket(**kwargs)

    def resolve_ticket(
        self,
        ticket: bytes,
        *,
        owner_subject: str,
    ) -> OutputRecord:
        return _output_record(self.ledger.resolve_ticket(
            ticket,
            owner_subject=owner_subject,
        ))


def _output_record(value: dict) -> OutputRecord:
    return OutputRecord(
        job_id=value["job_id"],
        ordinal=value["ordinal"],
        rows=value["rows"],
        batches=value["batches"],
        byte_count=value["bytes"],
        sha256=value["sha256"],
        schema_fingerprint=value["schema_fingerprint"],
        relative_path=value["relative_path"],
        published_at=value.get("published_at", 0.0),
    )


__all__ = ["PostgresOutputAccessStore"]
