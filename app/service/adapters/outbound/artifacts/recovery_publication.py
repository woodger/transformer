from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import BinaryIO, Protocol, cast

from app.contracts.checkpoint.v6 import CHECKPOINT_FORMAT, RECOVERY_FORMAT
from app.contracts.json_types import JsonObject
from app.contracts.worker.v12 import WorkerContractError, validate_document
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.ports.telemetry import TrainingTelemetryRepository
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
        input_revision: int,
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
    """Validate and register v6 checkpoints emitted by an active fit process."""

    def __init__(
        self,
        ledger: _RecoveryLedger,
        recovery_store: _RecoveryStore,
        spool: _AttemptSpool | None = None,
        *,
        telemetry: TrainingTelemetryRepository | None = None,
        logger: EventLogger,
        metrics: OperationalMetricSink,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ledger = ledger
        self.recovery_store = recovery_store
        self.spool = spool
        self.telemetry = telemetry
        self.logger = logger
        self.metrics = metrics
        self._monotonic = monotonic

    def publish(self, job: ExecutionJobRecord, event: JsonObject) -> None:
        started = self._monotonic()
        source_path, byte_count, digest = self._publish_artifact(job, event)
        generation = _positive_integer(event["generation"], "generation")
        progress = _object(event["progress"], "checkpoint progress")
        completed_epochs = _positive_integer(
            progress["completedEpochs"],
            "completed epochs",
        )
        global_step = _nonnegative_integer(
            progress["globalStep"],
            "global step",
        )
        training_complete = _boolean(
            progress["trainingComplete"],
            "training complete",
        )
        if generation != completed_epochs:
            raise WorkerRecoveryError(
                ErrorCode.MALFORMED_OUTPUT,
                "checkpoint generation differs from completed epochs",
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
        _, replayed = self.ledger.register_recovery_checkpoint(
            job_id=job.job_id,
            attempt=job.attempt,
            attempt_id=_attempt_id(job),
            generation=generation,
            input_revision=current.input_revision,
            format=RECOVERY_FORMAT,
            relative_path=self.recovery_store.relative_path(source_path),
            byte_count=byte_count,
            sha256=digest,
            completed_epochs=completed_epochs,
            global_step=global_step,
            training_complete=training_complete,
        )
        self._record_metrics(
            job,
            event,
            generation=generation,
            global_step=global_step,
            publication_ms=(self._monotonic() - started) * 1000.0,
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
        self._prune(job.job_id)

    def _publish_artifact(
        self,
        job: ExecutionJobRecord,
        event: JsonObject,
    ) -> tuple[str, int, str]:
        if self.spool is None:
            raise WorkerRecoveryError(
                ErrorCode.INTERNAL,
                "attempt spool is unavailable",
            )
        generation = _positive_integer(event["generation"], "generation")
        artifact = _object(event["artifact"], "checkpoint artifact")
        source = os.path.abspath(os.fspath(
            _string(artifact.get("path"), "checkpoint path")
        ))
        expected = self.spool.attempt_recovery_checkpoint_path(
            job.job_id,
            job.attempt,
            generation,
        )
        if artifact.get("format") != CHECKPOINT_FORMAT:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_INCOMPATIBLE,
                "worker checkpoint format is unsupported",
            )
        try:
            byte_count = os.path.getsize(source)
            digest = _sha256_file(source)
        except OSError as exc:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "worker checkpoint is unavailable",
            ) from exc
        if (
            source != expected
            or byte_count != _positive_integer(
                artifact.get("byteCount"),
                "checkpoint byteCount",
            )
            or digest != _string(
                artifact.get("checkpointSha256"),
                "checkpoint sha256",
            )
        ):
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "worker checkpoint integrity check failed",
            )
        destination = self.recovery_store.checkpoint_path(
            job.job_id,
            generation,
        )
        try:
            with open(source, "rb") as input_file:
                with self.recovery_store.staged_file(destination) as (output, _):
                    shutil.copyfileobj(input_file, output, _COPY_CHUNK_BYTES)
        except OSError as exc:
            raise WorkerRecoveryError(
                ErrorCode.RECOVERY_CHECKPOINT_UNAVAILABLE,
                "worker checkpoint could not be published",
            ) from exc
        return destination, byte_count, digest

    def _record_metrics(
        self,
        job: ExecutionJobRecord,
        event: JsonObject,
        *,
        generation: int,
        global_step: int,
        publication_ms: float,
    ) -> None:
        if "metrics" not in event or self.telemetry is None:
            return
        try:
            metrics = validate_document(event["metrics"], "training-metrics")
            serialization_ms = _nonnegative_number(
                event.get("checkpointSerializationMs"),
                "checkpoint serialization duration",
            )
            self.telemetry.record_epoch_interval(
                job_id=job.job_id,
                attempt=job.attempt,
                attempt_id=_attempt_id(job),
                generation=generation,
                global_step=global_step,
                metrics=metrics,
                checkpoint_serialization_ms=serialization_ms,
                checkpoint_publication_ms=publication_ms,
            )
        except (Exception, WorkerContractError) as exc:
            self.metrics.add("trainingTelemetryCollectionErrors")
            self.logger.event(
                "metrics.collection.failed",
                jobId=job.job_id,
                phase="recovery-checkpoint",
                errorType=type(exc).__name__,
            )

    def _prune(self, job_id: str) -> None:
        for relative_path in self.ledger.prune_recovery_checkpoints(job_id, keep=2):
            try:
                self.recovery_store.remove(
                    self.recovery_store.absolute_path(relative_path)
                )
            except OSError:
                self.logger.event(
                    "flight.recovery.cleanup_failed",
                    jobId=job_id,
                    artifact="checkpoint",
                )


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


def _nonnegative_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
    ):
        raise WorkerRecoveryError(
            ErrorCode.MALFORMED_OUTPUT,
            f"{label} must be a non-negative number",
        )
    return float(value)


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


__all__ = ["RecoveryCheckpointPublisher", "WorkerRecoveryError"]
