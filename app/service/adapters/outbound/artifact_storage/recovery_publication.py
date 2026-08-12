from __future__ import annotations

import hashlib
import os
import shutil
from contextlib import AbstractContextManager
from typing import BinaryIO, Protocol, cast

from app.contracts.json_types import JsonObject
from app.contracts.worker.v3.objective import TRAINING_RECOVERY_FORMAT
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.job import ErrorCode, InputState
from app.service.domain.records import (
    ExecutionJobRecord,
    TrainingRecoveryCheckpointRecord,
)

_COPY_CHUNK_BYTES = 1024 * 1024


class _RecoveryLedger(Protocol):
    def get_execution_job(self, job_id: str) -> ExecutionJobRecord | None: ...

    def register_recovery_checkpoint(
        self,
        *,
        job_id: str,
        attempt: int,
        attempt_id: str,
        generation: int,
        format: str,
        relative_path: str,
        byte_count: int,
        sha256: str,
        completed_epochs: int,
        global_step: int,
        training_complete: bool,
    ) -> tuple[TrainingRecoveryCheckpointRecord, bool]: ...

    def prune_recovery_checkpoints(
        self,
        job_id: str,
        *,
        keep: int = 2,
    ) -> list[str]: ...


class _RecoveryStore(Protocol):
    def checkpoint_path(self, job_id: str, generation: int) -> str: ...

    def relative_path(self, absolute_path: str) -> str: ...

    def absolute_path(self, relative_path: object) -> str: ...

    def staged_file(
        self,
        destination: str,
    ) -> AbstractContextManager[tuple[BinaryIO, str]]: ...

    def remove(self, path: str) -> bool: ...


class _AttemptSpool(Protocol):
    def attempt_recovery_checkpoint_path(
        self,
        job_id: str,
        attempt: int,
        generation: int,
    ) -> str: ...


class WorkerRecoveryError(AttemptExecutionError):
    pass


class RecoveryCheckpointPublisher:
    """Validate and register checkpoints emitted by an active fit process."""

    def __init__(
        self,
        ledger: _RecoveryLedger,
        recovery_store: _RecoveryStore,
        spool: _AttemptSpool | None = None,
        *,
        logger: EventLogger,
        metrics: OperationalMetricSink,
    ) -> None:
        self.ledger = ledger
        self.recovery_store = recovery_store
        self.spool = spool
        self.logger = logger
        self.metrics = metrics

    def publish(
        self,
        job: ExecutionJobRecord,
        event: JsonObject,
    ) -> None:
        if "artifact" in event:
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
        if set(event) != required:
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit subprocess emitted an invalid recovery event",
            )
        if event["format"] != TRAINING_RECOVERY_FORMAT:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_INCOMPATIBLE,
                "fit subprocess emitted an unsupported recovery checkpoint",
            )
        generation = _positive_integer(
            event["generation"],
            "recovery generation",
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
        if byte_count != _positive_integer(event["bytes"], "recovery bytes"):
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit recovery checkpoint size is invalid",
            )
        digest = _sha256_file(path)
        if digest != _string(event["sha256"], "recovery sha256"):
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "fit recovery checkpoint digest is invalid",
            )
        completed_epochs = _positive_integer(
            event["completed_epochs"],
            "completed epochs",
        )
        global_step = _nonnegative_integer(
            event["global_step"],
            "global step",
        )
        training_complete = _boolean(
            event["training_complete"],
            "training complete",
        )
        if completed_epochs != generation:
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "fit recovery checkpoint metadata is inconsistent",
            )
        _, replayed = self.ledger.register_recovery_checkpoint(
            job_id=job.job_id,
            attempt=job.attempt,
            attempt_id=_attempt_id(job),
            generation=generation,
            format=_string(event["format"], "recovery format"),
            relative_path=self.recovery_store.relative_path(path),
            byte_count=byte_count,
            sha256=digest,
            completed_epochs=completed_epochs,
            global_step=global_step,
            training_complete=training_complete,
        )
        if not replayed:
            self.metrics.add("recoveryCheckpointsPublished")
            self.metrics.add("recoveryCheckpointBytes", byte_count)
            self.logger.event(
                "flight.recovery.checkpoint",
                jobId=job.job_id,
                attempt=job.attempt,
                generation=generation,
                completedEpochs=completed_epochs,
                globalStep=global_step,
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

    def _publish_attempt_checkpoint(
        self,
        job: ExecutionJobRecord,
        event: JsonObject,
    ) -> JsonObject:
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
        generation = _positive_integer(
            event["generation"],
            "worker checkpoint generation",
        )
        artifact = _object(event["artifact"], "worker checkpoint artifact")
        source_path = os.path.abspath(os.fspath(
            _string(artifact.get("path"), "worker checkpoint path")
        ))
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
            or byte_count
            != _positive_integer(
                artifact.get("byteCount"),
                "worker checkpoint byteCount",
            )
            or _sha256_file(source_path)
            != _string(
                artifact.get("sha256"),
                "worker checkpoint sha256",
            )
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
            "completed_epochs": _positive_integer(
                event["completedEpochs"],
                "worker checkpoint completedEpochs",
            ),
            "global_step": _nonnegative_integer(
                event["globalStep"],
                "worker checkpoint globalStep",
            ),
            "training_complete": _boolean(
                event["trainingComplete"],
                "worker checkpoint trainingComplete",
            ),
            "bytes": byte_count,
            "sha256": _string(
                artifact.get("sha256"),
                "worker checkpoint sha256",
            ),
        }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise WorkerRecoveryError(
            ErrorCode.MALFORMED_OUTPUT,
            f"{label} must be an object",
        )
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise WorkerRecoveryError(
            ErrorCode.MALFORMED_OUTPUT,
            f"{label} keys must be strings",
        )
    return cast(JsonObject, dict(mapping))


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerRecoveryError(
            ErrorCode.MALFORMED_OUTPUT,
            f"{label} must be a non-empty string",
        )
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkerRecoveryError(
            ErrorCode.MALFORMED_OUTPUT,
            f"{label} must be a positive integer",
        )
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerRecoveryError(
            ErrorCode.MALFORMED_OUTPUT,
            f"{label} must be a non-negative integer",
        )
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise WorkerRecoveryError(
            ErrorCode.MALFORMED_OUTPUT,
            f"{label} must be a boolean",
        )
    return value


def _attempt_id(job: ExecutionJobRecord) -> str:
    if job.attempt_id is None:
        raise WorkerRecoveryError(
            ErrorCode.INTERNAL,
            "worker attempt identity is unavailable",
        )
    return job.attempt_id


__all__ = [
    "RecoveryCheckpointPublisher",
    "WorkerRecoveryError",
]
