from __future__ import annotations

import hashlib
import os

from app.service.application.ports.job_lifecycle import ArtifactLocation
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode


class ModelArtifactVerifier:
    def __init__(self, model_store):
        self.model_store = model_store

    def verify(self, model) -> None:
        try:
            path = self.model_store.model_absolute_path(
                model.checkpoint_path
            )
            expected = self.model_store.model_checkpoint_path(
                model.model_ref
            )
        except ValueError as exc:
            raise ServiceError(
                ErrorCode.MODEL_CORRUPT,
                "model checkpoint identity is invalid",
            ) from exc
        if path != expected:
            raise ServiceError(
                ErrorCode.MODEL_CORRUPT,
                "model checkpoint identity is invalid",
            )
        try:
            byte_count = os.path.getsize(path)
            checkpoint_sha256 = _sha256_file(path)
        except OSError as exc:
            raise ServiceError(
                ErrorCode.MODEL_UNAVAILABLE,
                "model checkpoint is unavailable",
            ) from exc
        if byte_count != model.byte_count or checkpoint_sha256 != model.sha256:
            raise ServiceError(
                ErrorCode.MODEL_CORRUPT,
                "model checkpoint integrity validation failed",
            )


class CandidateArtifactCleaner:
    def __init__(self, runtime_store, recovery_store, *, logger):
        self.runtime_store = runtime_store
        self.recovery_store = recovery_store
        self.logger = logger

    def cleanup(self, location: ArtifactLocation) -> None:
        store = (
            self.recovery_store
            if location.storage_class == "recovery"
            else self.runtime_store
        )
        try:
            store.remove(store.absolute_path(location.relative_path))
        except (FileNotFoundError, OSError):
            self.logger.event(
                "flight.input.candidate_cleanup_failed",
                storageClass=location.storage_class,
                path=os.path.basename(location.relative_path),
            )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["CandidateArtifactCleaner", "ModelArtifactVerifier"]
