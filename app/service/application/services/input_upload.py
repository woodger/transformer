from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import replace

from app.contracts.flight.v19.source_encoding import feature_block_dimensions
from app.contracts.worker.v17.constants import (
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
)
from app.service.application.messages.inputs import (
    CommittedInput,
    InputPayloadReceipt,
    InputUploadAuthorization,
    InputUploadJob,
    InputUploadMetadata,
)
from app.service.application.ports.input_uploads import InputUploadStore
from app.service.domain.errors import (
    ServiceError,
    conflict,
    failed_precondition,
    not_found,
    resource_exhausted,
)
from app.service.domain.job import ErrorCode, InputState
from app.service.domain.policies import resolve_device


class InputKindMismatch(Exception):
    pass


class InputUploadLifecycle:
    """Authorize and commit one physical payload around transport staging."""

    def __init__(
        self,
        store: InputUploadStore,
        *,
        max_payloads: int,
        max_job_bytes: int,
        recovery_enabled: bool,
        cuda_available: Callable[[], bool],
    ) -> None:
        self.store = store
        self.max_payloads = max_payloads
        self.max_job_bytes = max_job_bytes
        self.recovery_enabled = recovery_enabled
        self._cuda_available = cuda_available

    def authorize(
        self,
        owner: str,
        job_id: str,
        metadata: InputUploadMetadata,
    ) -> InputUploadAuthorization:
        self.validate_ordinal(metadata.ordinal)
        return self._authorize_job(owner, job_id, metadata)

    def validate_ordinal(self, ordinal: int) -> None:
        if ordinal >= self.max_payloads:
            raise resource_exhausted(
                f"input ordinal {ordinal} exceeds the per-job payload limit"
            )

    def _authorize_job(
        self,
        owner: str,
        job_id: str,
        metadata: InputUploadMetadata,
    ) -> InputUploadAuthorization:
        job = self.store.get_job(job_id, owner_subject=owner)
        if job is None:
            raise not_found("job not found")
        if job.input_state != InputState.OPEN:
            raise failed_precondition("job no longer accepts inputs")

        metadata = replace(
            metadata,
            schema_id=(
                FIT_INPUT_SCHEMA_ID
                if job.operation == "fit"
                else PREDICT_INPUT_SCHEMA_ID
            ),
            input_kind=job.operation,
            data_contract_sha256=job.data_contract_sha256,
        )
        _match_upload(job, metadata)
        _verify_fence(job, metadata)
        decision = resolve_device(
            job.requested_device,
            bool(self._cuda_available()),
        )
        if decision.error_code is not None:
            raise ServiceError(
                decision.error_code,
                "explicit GPU device is unavailable",
            )
        if decision.selected is None:
            raise ServiceError(ErrorCode.INTERNAL, "device selection failed")
        existing = self.store.find_input(
            job.job_id,
            ordinal=metadata.ordinal,
            payload_id=metadata.payload_id,
        )
        storage_class = (
            "recovery"
            if job.operation == "fit" and self.recovery_enabled
            else "runtime"
        )
        if existing is not None and existing.storage_class != storage_class:
            raise conflict("input is committed in a different storage class")
        return InputUploadAuthorization(
            job=job,
            metadata=metadata,
            selected_device=decision.selected,
            storage_class=storage_class,
            upload_token=secrets.token_hex(24),
            existing=existing,
        )

    def assert_accepting(
        self,
        authorization: InputUploadAuthorization,
    ) -> InputUploadJob:
        current = self.store.get_job(
            authorization.job.job_id,
            owner_subject=authorization.job.owner_subject,
        )
        if current is None:
            raise not_found("job not found")
        if current.input_state == InputState.ABORTED:
            raise ServiceError(ErrorCode.CANCELLED, "job input was aborted")
        if current.input_state != InputState.OPEN:
            raise failed_precondition("job no longer accepts inputs")
        _verify_fence(current, authorization.metadata)
        return current

    def reserve(
        self,
        authorization: InputUploadAuthorization,
        candidate_path: str,
    ) -> None:
        self.store.reserve(authorization, candidate_path)

    def commit(
        self,
        authorization: InputUploadAuthorization,
        receipt: InputPayloadReceipt,
    ) -> CommittedInput:
        return self.store.commit(
            authorization,
            receipt,
            max_payloads=self.max_payloads,
            max_job_bytes=self.max_job_bytes,
        )

    def replay(
        self,
        authorization: InputUploadAuthorization,
    ) -> CommittedInput:
        if authorization.existing is None:
            raise ValueError("committed input replay is unavailable")
        current = self.assert_accepting(authorization)
        return replace(
            authorization.existing,
            input_revision=current.input_revision,
            next_input_ordinal=current.next_input_ordinal,
            queued=False,
            frontier_advanced=False,
        )

    def abort(self, upload_token: str) -> str | None:
        return self.store.abort(upload_token)

    def find_committed(
        self,
        authorization: InputUploadAuthorization,
    ) -> CommittedInput | None:
        return self.store.find_input(
            authorization.job.job_id,
            ordinal=authorization.metadata.ordinal,
            payload_id=authorization.metadata.payload_id,
        )


def _match_upload(
    job: InputUploadJob,
    metadata: InputUploadMetadata,
) -> None:
    if metadata.input_kind != job.operation:
        raise InputKindMismatch
    if metadata.data_contract_sha256 != job.data_contract_sha256:
        raise ServiceError(
            ErrorCode.MODEL_SCHEMA_MISMATCH,
            "metadata dataContractSha256 does not match the job",
        )
    feature_dim = job.model_config.feature_dim
    blocks = feature_block_dimensions(
        job.source_encoding,
        feature_dim=feature_dim,
    )
    if len(metadata.native_rows) != len(blocks):
        raise failed_precondition(
            "metadata nativeRows does not match sourceEncoding"
        )


def _verify_fence(
    job: InputUploadJob,
    metadata: InputUploadMetadata,
) -> None:
    if (
        job.client_execution_id != metadata.client_execution_id
        or job.fencing_token != metadata.fencing_token
    ):
        raise ServiceError(
            ErrorCode.STALE_FENCE,
            "job ownership fence is stale",
        )


__all__ = ["InputKindMismatch", "InputUploadLifecycle"]
