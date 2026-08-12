from __future__ import annotations

from app.contracts.worker.v3.config import ModelConfig
from app.service.application.input_models import (
    CommittedInput,
    InputPayloadReceipt,
    InputUploadAuthorization,
    InputUploadJob,
)
from app.service.domain.errors import conflict, failed_precondition
from app.service.domain.job import InputState


class PostgresInputUploadStore:
    def __init__(self, ledger, *, schema_ids: dict[str, str]):
        self.ledger = ledger
        self.schema_ids = dict(schema_ids)

    def get_job(
        self,
        job_id: str,
        *,
        owner_subject: str,
    ) -> InputUploadJob | None:
        value = self.ledger.get_job(
            job_id,
            owner_subject=owner_subject,
        )
        if value is None:
            return None
        model_config = ModelConfig.from_dict(value["model_config"])
        if model_config is None:
            raise failed_precondition(
                "job model configuration is unavailable"
            )
        return InputUploadJob(
            job_id=value["job_id"],
            owner_subject=value["owner_subject"],
            operation=value["operation"],
            input_state=InputState(value["input_state"]),
            requested_device=value["requested_device"],
            selected_device=value["selected_device"],
            client_execution_id=value["client_execution_id"],
            fencing_token=value["fencing_token"],
            input_revision=value["input_revision"],
            next_input_ordinal=value["next_input_ordinal"],
            data_contract_sha256=value["data_contract_sha256"],
            model_config=model_config,
        )

    def find_input(
        self,
        job_id: str,
        *,
        ordinal: int,
        payload_id: str,
    ) -> CommittedInput | None:
        by_ordinal = self.ledger.find_input(job_id, ordinal=ordinal)
        by_payload = self.ledger.find_input(
            job_id,
            payload_id=payload_id,
        )
        if by_ordinal is None and by_payload is None:
            return None
        if by_ordinal is None or by_payload is None:
            raise conflict("input ordinal or payloadId is already committed")
        if by_ordinal["ordinal"] != by_payload["ordinal"]:
            raise conflict(
                "input ordinal and payloadId refer to different inputs"
            )
        return _committed_input(by_ordinal)

    def reserve(
        self,
        authorization: InputUploadAuthorization,
        candidate_path: str,
    ) -> None:
        metadata = authorization.metadata
        self.ledger.reserve_input(
            job_id=authorization.job.job_id,
            payload_id=metadata.payload_id,
            ordinal=metadata.ordinal,
            client_execution_id=metadata.client_execution_id,
            fencing_token=metadata.fencing_token,
            upload_token=authorization.upload_token,
            candidate_path=candidate_path,
            storage_class=authorization.storage_class,
        )

    def abort(self, upload_token: str) -> str | None:
        return self.ledger.abort_input(upload_token)

    def commit(
        self,
        authorization: InputUploadAuthorization,
        receipt: InputPayloadReceipt,
        *,
        max_payloads: int,
        max_job_bytes: int,
    ) -> CommittedInput:
        metadata = authorization.metadata
        if metadata.schema_id != self.schema_ids.get(metadata.input_kind):
            raise failed_precondition(
                f"schemaId {metadata.schema_id!r} does not match job operation"
            )
        value = self.ledger.commit_input(
            upload_token=authorization.upload_token,
            job_id=authorization.job.job_id,
            client_execution_id=metadata.client_execution_id,
            fencing_token=metadata.fencing_token,
            relative_path=receipt.relative_path,
            schema_id=metadata.schema_id,
            data_contract_sha256=metadata.data_contract_sha256,
            rows=receipt.rows,
            batches=receipt.batches,
            byte_count=receipt.byte_count,
            sha256=receipt.sha256,
            schema_fingerprint=receipt.schema_fingerprint,
            source_width=receipt.source_width,
            feature_dim=receipt.feature_dim,
            selected_device=authorization.selected_device,
            max_payloads=max_payloads,
            max_job_bytes=max_job_bytes,
            storage_class=authorization.storage_class,
        )
        return _committed_input(value)


def _committed_input(value: dict) -> CommittedInput:
    return CommittedInput(
        job_id=value["job_id"],
        payload_id=value["payload_id"],
        ordinal=value["ordinal"],
        schema_id=value["schema_id"],
        data_contract_sha256=value["data_contract_sha256"],
        rows=value["rows"],
        batches=value["batches"],
        byte_count=value["bytes"],
        sha256=value["sha256"],
        schema_fingerprint=value["schema_fingerprint"],
        relative_path=value["relative_path"],
        storage_class=value["storage_class"],
        source_width=value["source_width"],
        feature_dim=value["feature_dim"],
        input_revision=value.get("input_revision", 0),
        next_input_ordinal=value.get("next_input_ordinal", 0),
        queued=value.get("queued", False),
        frontier_advanced=value.get("frontier_advanced", False),
    )


__all__ = ["PostgresInputUploadStore"]
