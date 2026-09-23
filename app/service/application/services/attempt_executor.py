from __future__ import annotations

import errno
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from app.service.application.ports.artifacts import ArtifactPublisher
from app.service.application.ports.jobs import JobRepository
from app.service.application.ports.observability import (
    EventLogger,
    OperationalMetricSink,
)
from app.service.application.ports.telemetry import FitTelemetryPublisher
from app.service.application.ports.workers import (
    AttemptProcess,
    ExecutionPlanBuilder,
)
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode, ExecutionState
from app.service.domain.policies import decide_attempt_outcome
from app.service.domain.records import ExecutionJobRecord

_DISK_FULL_ERRNOS = {
    value
    for value in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
    if value is not None
}


@dataclass(frozen=True)
class WorkerAttemptError(Exception):
    code: ErrorCode
    message: str
    exit_code: int | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)


@dataclass
class _ActiveAttempt:
    job_id: str
    attempt: int
    cancel: threading.Event = field(default_factory=threading.Event)
    explicit_cancel: threading.Event = field(default_factory=threading.Event)


class WorkerAttemptExecutor:
    """Execute and durably finalize one already-claimed worker attempt."""

    def __init__(
        self,
        ledger: JobRepository,
        plan_builder: ExecutionPlanBuilder,
        subprocess_runner: AttemptProcess,
        artifact_publisher: ArtifactPublisher,
        *,
        fit_telemetry_publisher: FitTelemetryPublisher | None = None,
        logger: EventLogger,
        metrics: OperationalMetricSink,
        retry_notifier: Callable[[str], None] | None = None,
        confirm_device_loss: Callable[[str], bool] | None = None,
        resumable_fit: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ledger = ledger
        self.plan_builder = plan_builder
        self.subprocess_runner = subprocess_runner
        self.artifact_publisher = artifact_publisher
        self.fit_telemetry_publisher = fit_telemetry_publisher
        self.logger = logger
        self.metrics = metrics
        self.retry_notifier = retry_notifier
        self.confirm_device_loss = confirm_device_loss
        self.resumable_fit = bool(resumable_fit)
        self._monotonic = monotonic
        self._active_lock = threading.Lock()
        self._active: dict[str, _ActiveAttempt] = {}
        self._pending_explicit_cancellations: set[str] = set()
        self._force_stop = threading.Event()

    def notify_cancel(self, job_id: str) -> None:
        """Deliver a validated explicit cancellation to an active attempt."""
        with self._active_lock:
            active = self._active.get(job_id)
            if active is None:
                self._pending_explicit_cancellations.add(job_id)
            else:
                active.explicit_cancel.set()
                active.cancel.set()

    def notify_input(self, job_id: str) -> None:
        notifier = cast(
            object,
            getattr(self.subprocess_runner, "notify_input", None),
        )
        if notifier is not None:
            if not callable(notifier):
                raise TypeError("attempt input notifier must be callable")
            cast(Callable[[str], None], notifier)(job_id)

    def interrupt_for_shutdown(self) -> None:
        """Interrupt all attempts after the worker-pool drain deadline."""
        # Сначала устанавливаем принудительную остановку, чтобы попытка,
        # регистрируемая после этого снимка, не приняла остановку сервиса
        # за обычную отмену клиентом.
        self._force_stop.set()
        with self._active_lock:
            for active in self._active.values():
                active.cancel.set()

    def execute(self, job: ExecutionJobRecord) -> None:
        job_id = job.job_id
        attempt = job.attempt
        active = _ActiveAttempt(job_id, attempt)
        with self._active_lock:
            self._active[job_id] = active
            if job_id in self._pending_explicit_cancellations:
                self._pending_explicit_cancellations.remove(job_id)
                active.explicit_cancel.set()
                active.cancel.set()
            if self._force_stop.is_set():
                active.cancel.set()

        started = self._monotonic()
        succeeded = False
        try:
            current = self.ledger.get_execution_job(job_id)
            if current is None:
                raise WorkerAttemptError(
                    ErrorCode.INTERNAL,
                    "claimed job disappeared",
                )
            if not self._same_attempt(current, job):
                raise WorkerAttemptError(
                    ErrorCode.EXECUTION_INTERRUPTED,
                    "worker attempt no longer owns the job",
                )
            if current.execution_state == ExecutionState.CANCELLING:
                active.cancel.set()
                raise WorkerAttemptError(
                    ErrorCode.CANCELLED,
                    "job cancellation was requested",
                )
            if active.explicit_cancel.is_set():
                raise WorkerAttemptError(
                    ErrorCode.CANCELLED,
                    "job cancellation was requested",
                )
            if self._force_stop.is_set():
                raise WorkerAttemptError(
                    ErrorCode.EXECUTION_INTERRUPTED,
                    "worker execution was interrupted by service shutdown",
                )
            try:
                plan = self.plan_builder.build(job, attempt)
            except AttemptExecutionError as exc:
                raise WorkerAttemptError(exc.code, exc.message) from exc
            if active.explicit_cancel.is_set():
                raise WorkerAttemptError(
                    ErrorCode.CANCELLED,
                    "job cancellation was requested",
                )
            try:
                result = self.subprocess_runner.run(
                    job,
                    plan,
                    cancel=active.cancel,
                    force_stop=self._force_stop,
                )
            except AttemptExecutionError as exc:
                raise WorkerAttemptError(
                    exc.code,
                    exc.message,
                    exc.exit_code,
                ) from exc
            if active.explicit_cancel.is_set():
                raise WorkerAttemptError(
                    ErrorCode.CANCELLED,
                    "job cancellation was requested",
                )
            try:
                if job.operation == "predict":
                    closed = self.ledger.get_execution_job(job_id)
                    if closed is None or not self._same_attempt(closed, job):
                        raise WorkerAttemptError(
                            ErrorCode.EXECUTION_INTERRUPTED,
                            "worker attempt lost ownership before publication",
                        )
                    publication_inputs = self.plan_builder.streaming_inputs(
                        closed,
                        0,
                    )
                    self.artifact_publisher.publish_outputs_from_manifest(
                        job,
                        publication_inputs,
                        result.result_manifest,
                    )
                else:
                    closed = self.ledger.get_execution_job(job_id)
                    if closed is None or not self._same_attempt(closed, job):
                        raise WorkerAttemptError(
                            ErrorCode.EXECUTION_INTERRUPTED,
                            "worker attempt lost ownership before publication",
                        )
                    published_model = (
                        self.artifact_publisher.publish_model_from_manifest(
                            closed,
                            result.result_manifest,
                        )
                    )
                    if self.fit_telemetry_publisher is not None:
                        try:
                            self.fit_telemetry_publisher.publish(
                                closed,
                                result.result_manifest,
                                published_model,
                            )
                        except Exception as exc:
                            self.metrics.add(
                                "trainingTelemetryCollectionErrors"
                            )
                            self.logger.event(
                                "metrics.collection.failed",
                                jobId=job.job_id,
                                phase="model-artifact",
                                errorType=type(exc).__name__,
                            )
            except AttemptExecutionError as exc:
                raise WorkerAttemptError(exc.code, exc.message) from exc
            self._record_transition(
                job,
                ExecutionState.RUNNING.value,
                ExecutionState.SUCCEEDED.value,
            )
            succeeded = True
            self.metrics.add("jobsSucceeded")
        except BaseException as exc:
            current = self.ledger.get_execution_job(job_id)
            if current is not None and not self._same_attempt(current, job):
                self.artifact_publisher.cleanup_unpublished(job)
                self.logger.event(
                    "flight.worker.stale",
                    jobId=job_id,
                    attempt=attempt,
                    attemptId=job.attempt_id,
                )
                return
            if (
                current is not None
                and current.execution_state == ExecutionState.SUCCEEDED
            ):
                succeeded = True
            else:
                self.artifact_publisher.cleanup_unpublished(job)
                failure = self._coerce_failure(exc)
                if active.explicit_cancel.is_set():
                    failure = WorkerAttemptError(
                        ErrorCode.CANCELLED,
                        "job cancellation was requested",
                        failure.exit_code,
                    )
                elif failure.code == ErrorCode.DEVICE_LOST:
                    failure = self._confirm_device_loss(job, failure)
                if failure.code == ErrorCode.GPU_OUT_OF_MEMORY:
                    self.metrics.add("gpuOutOfMemory")
                    self.logger.event(
                        "flight.cuda.oom",
                        jobId=job_id,
                        attempt=attempt,
                        device=job.selected_device,
                    )
                elif failure.code == ErrorCode.DEVICE_LOST:
                    self.metrics.add("gpuUnavailableDuringExecution")
                # Очистка может пересечься с зафиксированной командой отмены.
                # Обновляем состояние до выбора итога, чтобы отмена побеждала,
                # когда была зарегистрирована до фиксации финального артефакта.
                current = self.ledger.get_execution_job(job_id)
                if current is not None and not self._same_attempt(current, job):
                    self.logger.event(
                        "flight.worker.stale",
                        jobId=job_id,
                        attempt=attempt,
                        attemptId=job.attempt_id,
                    )
                    return
                outcome = decide_attempt_outcome(
                    (
                        current.execution_state
                        if current is not None
                        else ExecutionState.RUNNING
                    ),
                    failure.code,
                    operation=job.operation,
                    resumable_fit=self.resumable_fit,
                )
                if outcome.target == ExecutionState.CANCELLED:
                    self._finish_cancelled(job, failure.exit_code)
                    self.metrics.add("jobsCancelled")
                elif outcome.target == ExecutionState.RETRYING:
                    if self._finish_retrying(job, failure):
                        self.metrics.add("jobsRetried")
                    elif (
                        (
                            latest := self.ledger.get_execution_job(job_id)
                        )
                        is not None
                        and latest.execution_state == ExecutionState.CANCELLED
                    ):
                        self.metrics.add("jobsCancelled")
                else:
                    self._finish_failed(job, failure)
                    self.metrics.add("jobsFailed")
                self.logger.event(
                    "flight.worker.failed",
                    jobId=job_id,
                    attempt=attempt,
                    code=failure.code.value,
                    errorType=type(exc).__name__,
                )
        finally:
            with self._active_lock:
                if self._active.get(job_id) is active:
                    self._active.pop(job_id, None)
                    self._pending_explicit_cancellations.discard(job_id)
            elapsed = self._monotonic() - started
            self.metrics.add("workerRunSeconds", elapsed)
            self.logger.event(
                "flight.worker.finished",
                jobId=job_id,
                attempt=attempt,
                device=job.selected_device,
                succeeded=succeeded,
                runSeconds=elapsed,
            )

    def _confirm_device_loss(
        self,
        job: ExecutionJobRecord,
        failure: WorkerAttemptError,
    ) -> WorkerAttemptError:
        if (
            job.selected_device != "cuda"
            or not job.assigned_device_id
            or self.confirm_device_loss is None
        ):
            return WorkerAttemptError(
                ErrorCode.SUBPROCESS_FAILED,
                "GPU subprocess failed without a verifiable device loss",
                failure.exit_code,
            )
        try:
            confirmed = self.confirm_device_loss(
                job.assigned_device_id
            )
        except Exception:
            confirmed = False
        if confirmed:
            return failure
        return WorkerAttemptError(
            ErrorCode.SUBPROCESS_FAILED,
            "GPU subprocess failed while its assigned device remained available",
            failure.exit_code,
        )

    def _finish_cancelled(
        self,
        job: ExecutionJobRecord,
        exit_code: int | None,
    ) -> None:
        current = self.ledger.get_execution_job(job.job_id)
        if current is None or not self._same_attempt(current, job):
            return
        if current.execution_state in (
            ExecutionState.CANCELLED,
            ExecutionState.SUCCEEDED,
            ExecutionState.FAILED,
        ):
            return
        transitioned = self.ledger.request_attempt_cancel(
            job.job_id,
            job.attempt,
            attempt_id=_attempt_id(job),
        )
        if transitioned:
            self._record_transition(
                job,
                ExecutionState.RUNNING.value,
                ExecutionState.CANCELLING.value,
            )
        self.ledger.finish_attempt(
            job.job_id,
            job.attempt,
            ExecutionState.CANCELLED,
            attempt_id=_attempt_id(job),
            exit_code=exit_code,
        )
        self._record_transition(
            job,
            ExecutionState.CANCELLING.value,
            ExecutionState.CANCELLED.value,
        )

    @staticmethod
    def _same_attempt(
        current: ExecutionJobRecord,
        claimed: ExecutionJobRecord,
    ) -> bool:
        return (
            claimed.attempt_id is not None
            and current.attempt == claimed.attempt
            and current.attempt_id == claimed.attempt_id
        )

    def _finish_failed(
        self,
        job: ExecutionJobRecord,
        failure: WorkerAttemptError,
    ) -> None:
        current = self.ledger.get_execution_job(job.job_id)
        if current is None or not self._same_attempt(current, job):
            return
        if current.execution_state in (
            ExecutionState.SUCCEEDED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        ):
            return
        if current.execution_state == ExecutionState.CANCELLING:
            self._finish_cancelled(job, failure.exit_code)
            return
        try:
            self.ledger.finish_attempt(
                job.job_id,
                job.attempt,
                ExecutionState.FAILED,
                attempt_id=_attempt_id(job),
                error_code=failure.code,
                error_message=failure.message,
                exit_code=failure.exit_code,
            )
            self._record_transition(
                job,
                ExecutionState.RUNNING.value,
                ExecutionState.FAILED.value,
                code=failure.code.value,
            )
        except (ServiceError, ValueError):
            # Если отмена сначала зафиксировала переход в PostgreSQL, попытка
            # перехода RUNNING -> FAILED обнаруживает CANCELLING. Перечитываем
            # состояние и завершаем отмену; иначе сохраняем исходную ошибку.
            latest = self.ledger.get_execution_job(job.job_id)
            if (
                latest is not None
                and self._same_attempt(latest, job)
                and latest.execution_state == ExecutionState.CANCELLING
            ):
                self._finish_cancelled(job, failure.exit_code)
                return
            if latest is not None and latest.execution_state in (
                ExecutionState.SUCCEEDED,
                ExecutionState.FAILED,
                ExecutionState.CANCELLED,
            ):
                return
            raise

    def _finish_retrying(
        self,
        job: ExecutionJobRecord,
        failure: WorkerAttemptError,
    ) -> bool:
        current = self.ledger.get_execution_job(job.job_id)
        if current is None or not self._same_attempt(current, job):
            return False
        if current.execution_state in (
            ExecutionState.SUCCEEDED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        ):
            return False
        if current.execution_state == ExecutionState.CANCELLING:
            self._finish_cancelled(job, failure.exit_code)
            return False
        try:
            self.ledger.schedule_retry(
                job.job_id,
                job.attempt,
                attempt_id=_attempt_id(job),
                error_code=failure.code,
                error_message=failure.message,
                exit_code=failure.exit_code,
            )
        except (ServiceError, ValueError):
            latest = self.ledger.get_execution_job(job.job_id)
            if (
                latest is not None
                and self._same_attempt(latest, job)
                and latest.execution_state == ExecutionState.CANCELLING
            ):
                self._finish_cancelled(job, failure.exit_code)
                return False
            if latest is not None and latest.execution_state in (
                ExecutionState.RETRYING,
                ExecutionState.SUCCEEDED,
                ExecutionState.FAILED,
                ExecutionState.CANCELLED,
            ):
                return (
                    self._same_attempt(latest, job)
                    and latest.execution_state == ExecutionState.RETRYING
                )
            raise
        self._record_transition(
            job,
            ExecutionState.RUNNING.value,
            ExecutionState.RETRYING.value,
            code=failure.code.value,
        )
        if self.retry_notifier is not None:
            self.retry_notifier(job.job_id)
        return True

    def _record_transition(
        self,
        job: ExecutionJobRecord,
        from_state: str,
        to_state: str,
        *,
        code: str | None = None,
    ) -> None:
        self.metrics.record_transition(from_state, to_state)
        fields = {
            "jobId": job.job_id,
            "attempt": job.attempt,
            "device": job.selected_device,
            "fromState": from_state,
            "toState": to_state,
        }
        if code is not None:
            fields["code"] = code
        self.logger.event("flight.job.transition", **fields)

    @staticmethod
    def _coerce_failure(exc: BaseException) -> WorkerAttemptError:
        if isinstance(exc, WorkerAttemptError):
            return exc
        if isinstance(exc, ServiceError):
            return WorkerAttemptError(exc.code, exc.message)
        if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
            return WorkerAttemptError(
                ErrorCode.DISK_FULL,
                "runtime directory is full",
            )
        return WorkerAttemptError(
            ErrorCode.INTERNAL,
            "worker execution failed internally",
        )


def _attempt_id(job: ExecutionJobRecord) -> str:
    if job.attempt_id is None:
        raise ValueError("claimed worker job has no attempt identity")
    return job.attempt_id
