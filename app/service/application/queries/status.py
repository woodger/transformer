from __future__ import annotations

from app.service.application.job_models import (
    DescribeModelQuery,
    GetJobStatusQuery,
    JobInputsPage,
    JobOutputsPage,
    JobStatusResult,
    ListJobInputsQuery,
    ListJobOutputsQuery,
    ModelDescription,
)
from app.service.application.ports.job_lifecycle import ModelArtifactVerifier
from app.service.application.ports.job_queries import JobQueryStore
from app.service.application.services.model_contract import (
    verify_model_semantics,
)
from app.service.domain.errors import ServiceError, not_found
from app.service.domain.job import ErrorCode


class GetJobStatus:
    """Read one bounded, consistent durable status snapshot."""

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


class DescribeModel:
    def __init__(
        self,
        store: JobQueryStore,
        *,
        model_verifier: ModelArtifactVerifier,
    ) -> None:
        self.store = store
        self._model_verifier = model_verifier

    def execute(self, query: DescribeModelQuery) -> ModelDescription:
        if query.model_selector == "alias":
            model = self.store.resolve_published_model_alias(
                query.owner_subject,
                query.model_ref,
            )
        else:
            model = self.store.get_published_model(
                query.model_ref,
                owner_subject=query.owner_subject,
            )
        if model is None:
            raise not_found("model generation not found")
        model_config = verify_model_semantics(model)
        self._model_verifier.verify(model)
        return ModelDescription(query.request_id, model, model_config)

__all__ = [
    "DescribeModel",
    "GetJobStatus",
    "ListJobInputs",
    "ListJobOutputs",
]
