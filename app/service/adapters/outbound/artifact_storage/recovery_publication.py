from __future__ import annotations

import hashlib
import os
import shutil

from app.contracts.worker.v2.config import TRAINING_RECOVERY_FORMAT
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.job import ErrorCode, InputState

_COPY_CHUNK_BYTES = 1024 * 1024


class WorkerRecoveryError(AttemptExecutionError):
    pass


class RecoveryCheckpointPublisher:
    """Validate and register checkpoints emitted by an active fit process."""

    def __init__(
        self,
        ledger,
        recovery_store,
        spool=None,
        *,
        logger,
        metrics,
    ):
        self.ledger = ledger
        self.recovery_store = recovery_store
        self.spool = spool
        self.logger = logger
        self.metrics = metrics

    def publish(self, job, event: dict) -> None:
        if isinstance(event, dict) and "artifact" in event:
            event = self._publish_attempt_checkpoint(job, event)
        required = {
            "format",
            "generation",
            "completed_epochs",
            "global_step",
            "training_complete",
            "bytes",
            "sha256",
        }
        if not isinstance(event, dict) or set(event) != required:
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit subprocess emitted an invalid recovery event",
            )
        if event["format"] != TRAINING_RECOVERY_FORMAT:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_INCOMPATIBLE,
                "fit subprocess emitted an unsupported recovery checkpoint",
            )
        generation = event["generation"]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
        ):
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit subprocess emitted an invalid recovery generation",
            )
        current = self.ledger.get_execution_job(job.job_id)
        if (
            current is None
            or current.attempt != job.attempt
            or current.attempt_id != job.attempt_id
            or current.input_state is not InputState.CLOSED
            or current.manifest_sha256 is None
        ):
            raise WorkerRecoveryError(
                ErrorCode.INTERNAL,
                "closed fit input manifest is unavailable",
            )
        path = self.recovery_store.checkpoint_path(
            job.job_id,
            generation,
        )
        try:
            byte_count = os.path.getsize(path)
        except OSError as exc:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit recovery checkpoint is unavailable",
            ) from exc
        if byte_count != event["bytes"]:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit recovery checkpoint size is invalid",
            )
        digest = _sha256_file(path)
        if digest != event["sha256"]:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit recovery checkpoint digest is invalid",
            )
        metadata = {
            "generation": generation,
            "completed_epochs": event["completed_epochs"],
            "global_step": event["global_step"],
            "training_complete": event["training_complete"],
        }
        if (
            metadata["completed_epochs"] != generation
            or isinstance(metadata["global_step"], bool)
            or not isinstance(metadata["global_step"], int)
            or metadata["global_step"] < 0
            or not isinstance(metadata["training_complete"], bool)
        ):
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit recovery checkpoint metadata is inconsistent",
            )
        _, replayed = self.ledger.register_recovery_checkpoint(
            job_id=job.job_id,
            attempt=job.attempt,
            attempt_id=job.attempt_id,
            generation=generation,
            format=event["format"],
            relative_path=self.recovery_store.relative_path(path),
            byte_count=byte_count,
            sha256=digest,
            completed_epochs=event["completed_epochs"],
            global_step=event["global_step"],
            training_complete=event["training_complete"],
        )
        if not replayed:
            self.metrics.add("recoveryCheckpointsPublished")
            self.metrics.add("recoveryCheckpointBytes", byte_count)
            self.logger.event(
                "flight.recovery.checkpoint",
                jobId=job.job_id,
                attempt=job.attempt,
                generation=generation,
                completedEpochs=event["completed_epochs"],
                globalStep=event["global_step"],
                bytes=byte_count,
            )
        for relative_path in self.ledger.prune_recovery_checkpoints(
            job.job_id,
            keep=2,
        ):
            try:
                self.recovery_store.remove(
                    self.recovery_store.absolute_path(relative_path)
                )
            except OSError:
                self.logger.event(
                    "flight.recovery.cleanup_failed",
                    jobId=job.job_id,
                    artifact="checkpoint",
                )

    def _publish_attempt_checkpoint(self, job, event: dict) -> dict:
        required = {
            "generation",
            "completedEpochs",
            "globalStep",
            "trainingComplete",
            "artifact",
        }
        if set(event) != required or self.spool is None:
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit worker emitted an invalid checkpoint event",
            )
        generation = event["generation"]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
        ):
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit worker emitted an invalid checkpoint generation",
            )
        artifact = event["artifact"]
        source_path = os.path.abspath(os.fspath(artifact["path"]))
        expected_source = self.spool.attempt_recovery_checkpoint_path(
            job.job_id,
            job.attempt,
            generation,
        )
        try:
            byte_count = os.path.getsize(source_path)
        except OSError as exc:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit worker checkpoint is unavailable",
            ) from exc
        if (
            source_path != expected_source
            or byte_count != artifact["byteCount"]
            or _sha256_file(source_path) != artifact["sha256"]
        ):
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit worker checkpoint integrity check failed",
            )
        destination = self.recovery_store.checkpoint_path(
            job.job_id,
            generation,
        )
        try:
            with open(source_path, "rb") as source:
                with self.recovery_store.staged_file(destination) as (
                    target,
                    _,
                ):
                    shutil.copyfileobj(source, target, _COPY_CHUNK_BYTES)
        except OSError as exc:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit worker checkpoint could not be published",
            ) from exc
        return {
            "format": TRAINING_RECOVERY_FORMAT,
            "generation": generation,
            "completed_epochs": event["completedEpochs"],
            "global_step": event["globalStep"],
            "training_complete": event["trainingComplete"],
            "bytes": byte_count,
            "sha256": artifact["sha256"],
        }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "RecoveryCheckpointPublisher",
    "WorkerRecoveryError",
]
