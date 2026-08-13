from __future__ import annotations

from typing import Protocol

from app.service.application.messages.jobs import StoredInputPage, StoredOutputPage
from app.service.domain.records import PublishedModelRecord, StatusSnapshot


class JobQueryStore(Protocol):
    """Read models and bounded job projections for application queries."""

    def get_status_snapshot_record(
        self,
        job_id: str,
        owner_subject: str,
    ) -> StatusSnapshot: ...

    def is_job_retired(
        self,
        job_id: str,
        *,
        owner_subject: str,
    ) -> bool: ...

    def list_inputs_page(
        self,
        job_id: str,
        owner_subject: str,
        *,
        after_revision: int,
        snapshot_revision: int | None,
        cursor: int | None,
        limit: int,
    ) -> StoredInputPage: ...

    def list_outputs_page(
        self,
        job_id: str,
        owner_subject: str,
        *,
        cursor: int | None,
        limit: int,
    ) -> StoredOutputPage: ...

    def get_published_model(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
    ) -> PublishedModelRecord | None: ...

    def resolve_published_model_alias(
        self,
        owner_subject: str,
        label: str,
    ) -> PublishedModelRecord | None: ...


__all__ = ["JobQueryStore"]
