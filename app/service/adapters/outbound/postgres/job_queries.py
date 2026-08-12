from __future__ import annotations

from app.service.application.job_models import StoredInputPage, StoredOutputPage
from app.service.domain.records import (
    InputRecord,
    OutputRecord,
    PublishedModelRecord,
    StatusSnapshot,
)


class PostgresJobQueryStore:
    def __init__(self, ledger):
        self.ledger = ledger

    def get_status_snapshot_record(
        self,
        job_id: str,
        owner_subject: str,
    ) -> StatusSnapshot:
        return self.ledger.get_status_snapshot_record(job_id, owner_subject)

    def is_job_retired(
        self,
        job_id: str,
        *,
        owner_subject: str,
    ) -> bool:
        identity = self.ledger.get_job_identity(
            job_id,
            owner_subject=owner_subject,
        )
        return identity is not None and identity["retired_at"] is not None

    def list_inputs_page(
        self,
        job_id: str,
        owner_subject: str,
        *,
        after_revision: int,
        snapshot_revision: int | None,
        cursor: int | None,
        limit: int,
    ) -> StoredInputPage:
        page = self.ledger.list_inputs_page(
            job_id,
            owner_subject,
            after_revision=after_revision,
            snapshot_revision=snapshot_revision,
            cursor=cursor,
            limit=limit,
        )
        return StoredInputPage(
            after_revision=page["after_revision"],
            snapshot_revision=page["snapshot_revision"],
            cursor=page["cursor"],
            items=tuple(_input_record(item) for item in page["items"]),
            next_cursor=page["next_cursor"],
            has_more=page["has_more"],
        )

    def list_outputs_page(
        self,
        job_id: str,
        owner_subject: str,
        *,
        cursor: int | None,
        limit: int,
    ) -> StoredOutputPage:
        page = self.ledger.list_outputs_page(
            job_id,
            owner_subject,
            cursor=cursor,
            limit=limit,
        )
        return StoredOutputPage(
            cursor=page["cursor"],
            items=tuple(_output_record(item) for item in page["items"]),
            next_cursor=page["next_cursor"],
            has_more=page["has_more"],
        )

    def get_published_model(
        self,
        model_ref: str,
        *,
        owner_subject: str | None = None,
    ) -> PublishedModelRecord | None:
        return self.ledger.get_published_model(
            model_ref,
            owner_subject=owner_subject,
        )

    def resolve_published_model_alias(
        self,
        owner_subject: str,
        label: str,
    ) -> PublishedModelRecord | None:
        return self.ledger.resolve_published_model_alias(
            owner_subject,
            label,
        )


def _input_record(item: dict) -> InputRecord:
    return InputRecord(
        job_id=item["job_id"],
        ordinal=item["ordinal"],
        payload_id=item["payload_id"],
        commit_revision=item["commit_revision"],
        schema_id=item["schema_id"],
        data_contract_sha256=item["data_contract_sha256"],
        rows=item["rows"],
        batches=item["batches"],
        byte_count=item["bytes"],
        sha256=item["sha256"],
        schema_fingerprint=item["schema_fingerprint"],
        relative_path=item["relative_path"],
        storage_class=item["storage_class"],
        source_width=item["source_width"],
        feature_dim=item["feature_dim"],
        committed_at=item["committed_at"],
    )


def _output_record(item: dict) -> OutputRecord:
    return OutputRecord(
        job_id=item["job_id"],
        ordinal=item["ordinal"],
        rows=item["rows"],
        batches=item["batches"],
        byte_count=item["bytes"],
        sha256=item["sha256"],
        schema_fingerprint=item["schema_fingerprint"],
        relative_path=item["relative_path"],
        published_at=item["published_at"],
    )


__all__ = ["PostgresJobQueryStore"]
