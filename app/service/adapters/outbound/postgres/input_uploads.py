from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from app.contracts.worker.v8.config import ModelConfig
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.mapping import (
    row_integer,
    row_optional_string,
    row_string,
)
from app.service.application.messages.inputs import (
    CommittedInput,
    InputPayloadReceipt,
    InputUploadAuthorization,
    InputUploadJob,
)
from app.service.domain.errors import conflict, failed_precondition
from app.service.domain.job import InputState
from app.service.domain.json_types import JsonObject


class PostgresInputUploadStore:
    def __init__(
        self,
        ledger: Ledger,
        *,
        schema_ids: dict[str, str],
    ) -> None:
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
        raw_ml_contract = value["ml_contract"]
        if not isinstance(raw_ml_contract, Mapping):
            raise failed_precondition("job ML contract is unavailable")
        ml_contract = cast(
            JsonObject,
            dict(cast(Mapping[str, object], raw_ml_contract)),
        )
        return InputUploadJob(
            job_id=row_string(value, "job_id"),
            owner_subject=row_string(value, "owner_subject"),
            operation=row_string(value, "operation"),
            input_state=InputState(row_string(value, "input_state")),
            requested_device=row_string(value, "requested_device"),
            selected_device=row_optional_string(value, "selected_device"),
            client_execution_id=row_string(value, "client_execution_id"),
            fencing_token=row_integer(value, "fencing_token"),
            input_revision=row_integer(value, "input_revision"),
            next_input_ordinal=row_integer(value, "next_input_ordinal"),
            data_contract_sha256=row_string(value, "data_contract_sha256"),
            ml_contract=ml_contract,
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


def _committed_input(value: Mapping[str, object]) -> CommittedInput:
    return CommittedInput(
        job_id=row_string(value, "job_id"),
        payload_id=row_string(value, "payload_id"),
        ordinal=row_integer(value, "ordinal"),
        schema_id=row_string(value, "schema_id"),
        data_contract_sha256=row_string(value, "data_contract_sha256"),
        rows=row_integer(value, "rows"),
        batches=row_integer(value, "batches"),
        byte_count=row_integer(value, "bytes"),
        sha256=row_string(value, "sha256"),
        schema_fingerprint=row_string(value, "schema_fingerprint"),
        relative_path=row_string(value, "relative_path"),
        storage_class=row_string(value, "storage_class"),
        source_width=row_integer(value, "source_width"),
        feature_dim=row_integer(value, "feature_dim"),
        input_revision=_integer_or_default(value, "input_revision", 0),
        next_input_ordinal=_integer_or_default(
            value,
            "next_input_ordinal",
            0,
        ),
        queued=_boolean_or_default(value, "queued", False),
        frontier_advanced=_boolean_or_default(
            value,
            "frontier_advanced",
            False,
        ),
    )


def _integer_or_default(
    value: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    return default if key not in value else row_integer(value, key)


def _boolean_or_default(
    value: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    item = value.get(key, default)
    if not isinstance(item, bool):
        raise ValueError(f"database field {key} must be a boolean")
    return item


__all__ = ["PostgresInputUploadStore"]
