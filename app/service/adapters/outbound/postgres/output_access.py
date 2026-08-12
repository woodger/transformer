from __future__ import annotations

from collections.abc import Mapping

from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.mapping import (
    row_integer,
    row_optional_float,
    row_string,
)
from app.service.domain.job import ExecutionState
from app.service.domain.records import OutputRecord


class PostgresOutputAccessStore:
    def __init__(self, ledger: Ledger) -> None:
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
            else ExecutionState(row_string(job, "execution_state"))
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
                if row_integer(item, "ordinal") == ordinal
            ),
            None,
        )
        return None if value is None else _output_record(value)

    def issue_ticket(
        self,
        *,
        job_id: str,
        ordinal: int,
        owner_subject: str,
        ttl_seconds: float,
    ) -> tuple[bytes, float]:
        return self.ledger.issue_ticket(
            job_id=job_id,
            ordinal=ordinal,
            owner_subject=owner_subject,
            ttl_seconds=ttl_seconds,
        )

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


def _output_record(value: Mapping[str, object]) -> OutputRecord:
    return OutputRecord(
        job_id=row_string(value, "job_id"),
        ordinal=row_integer(value, "ordinal"),
        rows=row_integer(value, "rows"),
        batches=row_integer(value, "batches"),
        byte_count=row_integer(value, "bytes"),
        sha256=row_string(value, "sha256"),
        schema_fingerprint=row_string(value, "schema_fingerprint"),
        relative_path=row_string(value, "relative_path"),
        published_at=row_optional_float(value, "published_at") or 0.0,
    )


__all__ = ["PostgresOutputAccessStore"]
