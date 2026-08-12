from __future__ import annotations

from typing import Protocol

from app.service.application.input_models import (
    CommittedInput,
    InputPayloadReceipt,
    InputUploadAuthorization,
    InputUploadJob,
)


class InputUploadStore(Protocol):
    def get_job(
        self,
        job_id: str,
        *,
        owner_subject: str,
    ) -> InputUploadJob | None: ...

    def find_input(
        self,
        job_id: str,
        *,
        ordinal: int,
        payload_id: str,
    ) -> CommittedInput | None: ...

    def reserve(
        self,
        authorization: InputUploadAuthorization,
        candidate_path: str,
    ) -> None: ...

    def abort(self, upload_token: str) -> str | None: ...

    def commit(
        self,
        authorization: InputUploadAuthorization,
        receipt: InputPayloadReceipt,
        *,
        max_payloads: int,
        max_job_bytes: int,
    ) -> CommittedInput: ...


class InputArtifactStore(Protocol):
    def input_candidate_path(
        self,
        job_id: str,
        ordinal: int,
        payload_id: str,
        upload_token: str,
    ) -> str: ...

    def create_temporary(self, destination: str): ...

    def durable_create(self, temporary: str, destination: str) -> None: ...

    def relative_path(self, path: str) -> str: ...

    def absolute_path(self, relative_path: str) -> str: ...

    def remove(self, path: str) -> bool: ...


__all__ = ["InputArtifactStore", "InputUploadStore"]
