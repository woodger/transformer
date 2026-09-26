from __future__ import annotations

from app.service.application.messages.jobs import (
    GetJobStatusQuery,
    JobInputsPage,
    JobOutputsPage,
    JobStatusResult,
    ListJobInputsQuery,
    ListJobOutputsQuery,
)
from app.service.application.ports.job_queries import JobQueryStore
from app.service.domain.errors import ServiceError, not_found
from app.service.domain.job import ErrorCode


class GetJobStatus:
    """Прочитать один ограниченный согласованный долговременный снимок статуса."""

    def __init__(self, store: JobQueryStore) -> None:
        self.store = store

    def execute(self, query: GetJobStatusQuery) -> JobStatusResult:
        snapshot = self.store.get_status_snapshot_record(
            query.job_id,
            query.owner_subject,
        )
        if snapshot.job is None:
            retired = self.store.is_job_retired(
                query.job_id,
                owner_subject=query.owner_subject,
            )
            if retired:
                raise ServiceError(
                    ErrorCode.JOB_RETIRED,
                    "job identity has been retired",
                )
            raise not_found("job not found")
        return JobStatusResult(query.request_id, snapshot)


class ListJobInputs:
    def __init__(self, store: JobQueryStore) -> None:
        self.store = store

    def execute(self, query: ListJobInputsQuery) -> JobInputsPage:
        page = self.store.list_inputs_page(
            query.job_id,
            query.owner_subject,
            after_revision=query.after_revision,
            snapshot_revision=query.snapshot_revision,
            cursor=query.cursor,
            limit=query.limit,
        )
        return JobInputsPage(
            request_id=query.request_id,
            job_id=query.job_id,
            after_revision=page.after_revision,
            snapshot_revision=page.snapshot_revision,
            cursor=page.cursor,
            items=page.items,
            next_cursor=page.next_cursor,
            has_more=page.has_more,
        )


class ListJobOutputs:
    def __init__(self, store: JobQueryStore) -> None:
        self.store = store

    def execute(self, query: ListJobOutputsQuery) -> JobOutputsPage:
        page = self.store.list_outputs_page(
            query.job_id,
            query.owner_subject,
            cursor=query.cursor,
            limit=query.limit,
        )
        return JobOutputsPage(
            request_id=query.request_id,
            job_id=query.job_id,
            cursor=page.cursor,
            items=page.items,
            next_cursor=page.next_cursor,
            has_more=page.has_more,
        )


__all__ = [
    "GetJobStatus",
    "ListJobInputs",
    "ListJobOutputs",
]
