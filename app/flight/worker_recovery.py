from __future__ import annotations

import os
from dataclasses import dataclass

from app.flight.constants import ErrorCode
from app.storage.training_recovery import (
    TRAINING_RECOVERY_FORMAT,
    load_training_recovery,
    sha256_file,
)


@dataclass(frozen=True)
class WorkerRecoveryError(Exception):
    code: ErrorCode
    message: str

    def __post_init__(self):
        Exception.__init__(self, self.message)


class RecoveryCheckpointPublisher:
    """Validate and register checkpoints emitted by an active fit process."""

    def __init__(
        self,
        ledger,
        recovery_store,
        *,
        logger,
        metrics,
    ):
        self.ledger = ledger
        self.recovery_store = recovery_store
        self.logger = logger
        self.metrics = metrics

    def publish(self, job, event: dict) -> None:
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
        if job.seal_hash is None:
            raise WorkerRecoveryError(
                ErrorCode.INTERNAL,
                "fit job seal hash is unavailable",
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
        digest = sha256_file(path)
        if digest != event["sha256"]:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit recovery checkpoint digest is invalid",
            )
        try:
            payload = load_training_recovery(
                path,
                "cpu",
                expected_config_hash=job.config_hash,
                expected_seal_hash=job.seal_hash,
            )
        except Exception as exc:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_INCOMPATIBLE,
                "fit recovery checkpoint is invalid",
            ) from exc
        metadata = {
            "generation": payload["generation"],
            "completed_epochs": payload["completed_epochs"],
            "global_step": payload["global_step"],
            "training_complete": payload["training_complete"],
        }
        if any(
            event[name] != value
            for name, value in metadata.items()
        ):
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit recovery event does not match its checkpoint",
            )
        _, replayed = self.ledger.register_recovery_checkpoint(
            job_id=job.job_id,
            attempt=job.attempt,
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


__all__ = [
    "RecoveryCheckpointPublisher",
    "WorkerRecoveryError",
]
