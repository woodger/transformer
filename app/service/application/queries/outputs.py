from __future__ import annotations

from app.service.application.messages.outputs import OutputTicketGrant
from app.service.application.ports.output_access import OutputAccessStore
from app.service.domain.errors import failed_precondition, not_found
from app.service.domain.job import ExecutionState
from app.service.domain.records import OutputRecord


class OutputAccess:
    """Авторизовать поиск опубликованного вывода и выдачу билета."""

    def __init__(
        self,
        store: OutputAccessStore,
        *,
        ticket_ttl_seconds: float,
    ) -> None:
        self.store = store
        self.ticket_ttl_seconds = ticket_ttl_seconds

    def locate(
        self,
        owner: str,
        job_id: str,
        ordinal: int,
    ) -> OutputRecord:
        state = self.store.execution_state(
            job_id,
            owner_subject=owner,
        )
        if state is None:
            raise not_found("job output not found")
        if state != ExecutionState.SUCCEEDED:
            raise failed_precondition(
                "job outputs are available only after successful execution"
            )
        output = self.store.find_output(job_id, ordinal)
        if output is None:
            raise not_found("job output not found")
        return output

    def issue(
        self,
        owner: str,
        job_id: str,
        ordinal: int,
    ) -> OutputTicketGrant:
        return self.store.issue_ticket(
            job_id=job_id,
            ordinal=ordinal,
            owner_subject=owner,
            ttl_seconds=self.ticket_ttl_seconds,
        )

    def resolve(self, owner: str, token: bytes) -> OutputRecord:
        return self.store.resolve_ticket(token, owner_subject=owner)


__all__ = ["OutputAccess"]
