from __future__ import annotations

import errno
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.service.application.ports.artifacts import ArtifactPublisher
from app.service.application.ports.jobs import JobRepository
from app.service.application.ports.workers import (
    AttemptProcess,
    ExecutionPlanBuilder,
)
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode, JobState
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

    def __post_init__(self):
        Exception.__init__(self, self.message)


@dataclass
class _ActiveAttempt:
    job_id: str
    attempt: int
    cancel: threading.Event = field(default_factory=threading.Event)


class WorkerAttemptExecutor:
    """Execute and durably finalize one already-claimed worker attempt."""

    def __init__(
        self,
        ledger: JobRepository,
        plan_builder: ExecutionPlanBuilder,
        subprocess_runner: AttemptProcess,
        artifact_publisher: ArtifactPublisher,
        *,
        logger,
        metrics,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
        retry_notifier: Callable[[str], None] | None = None,
        confirm_device_loss: Callable[[str], bool] | None = None,
        resumable_fit: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.ledger = ledger
        self.plan_builder = plan_builder
        self.subprocess_runner = subprocess_runner
        self.artifact_publisher = artifact_publisher
        self.logger = logger
        self.metrics = metrics
        self.argv_hook = argv_hook
        self.retry_notifier = retry_notifier
        self.confirm_device_loss = confirm_device_loss
        self.resumable_fit = bool(resumable_fit)
        self._monotonic = monotonic
        self._active_lock = threading.Lock()
        self._active: dict[str, _ActiveAttempt] = {}
        self._pending_cancellations: set[str] = set()
        self._force_stop = threading.Event()

    def notify_cancel(self, job_id: str) -> None:
        """Deliver a committed RUNNING -> CANCELLING notification."""
        with self._active_lock:
            active = self._active.get(job_id)
            if active is None:
                self._pending_cancellations.add(job_id)
            else:
                active.cancel.set()

    def interrupt_for_shutdown(self) -> None:
        """Interrupt all attempts after the worker-pool drain deadline."""
        # Set force-stop first so an attempt registering after this snapshot
        # cannot mistake shutdown for an ordinary client cancellation.
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
            if job_id in self._pending_cancellations:
                self._pending_cancellations.remove(job_id)
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
            if current.state == JobState.CANCELLING:
                active.cancel.set()
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
                plan = self.plan_builder.build(
                    job,
                    attempt,
                    argv_hook=self.argv_hook,
                )
            except AttemptExecutionError as exc:
                raise WorkerAttemptError(exc.code, exc.message) from exc
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
            try:
                if job.operation == "predict":
                    if result.result_manifest is None:
                        self.artifact_publisher.publish_outputs(
                            job,
                            plan.inputs,
                            result.outputs,
                        )
                    else:
                        self.artifact_publisher.publish_outputs_from_manifest(
                            job,
                            plan.inputs,
                            result.result_manifest,
                        )
                else:
                    if result.result_manifest is None:
                        self.artifact_publisher.publish_model(job)
                    else:
                        self.artifact_publisher.publish_model_from_manifest(
                            job,
                            result.result_manifest,
                        )
            except AttemptExecutionError as exc:
                raise WorkerAttemptError(exc.code, exc.message) from exc
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.SUCCEEDED.value,
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
            if current is not None and current.state == JobState.SUCCEEDED:
                succeeded = True
            else:
                self.artifact_publisher.cleanup_unpublished(job)
                failure = self._coerce_failure(exc)
                if failure.code == ErrorCode.DEVICE_LOST:
                    failure = self._confirm_device_loss(job, failure)
                if failure.code == ErrorCode.CUDA_OUT_OF_MEMORY:
                    self.metrics.add("cudaOutOfMemory")
                    self.logger.event(
                        "flight.cuda.oom",
                        jobId=job_id,
                        attempt=attempt,
                        device=job.selected_device,
                    )
                elif failure.code == ErrorCode.DEVICE_LOST:
                    self.metrics.add("cudaUnavailableDuringExecution")
                # Cleanup can race with a committed cancel action. Refresh
                # state before choosing the terminal outcome so cancel wins
                # whenever it was registered before final artifact commit.
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
                        current.state
                        if current is not None
                        else JobState.RUNNING
                    ),
                    failure.code,
                    operation=job.operation,
                    resumable_fit=self.resumable_fit,
                )
                if outcome.target == JobState.CANCELLED:
                    self._finish_cancelled(job, failure.exit_code)
                    self.metrics.add("jobsCancelled")
                elif outcome.target == JobState.RETRYING:
                    if self._finish_retrying(job, failure):
                        self.metrics.add("jobsRetried")
                    elif (
                        (
                            latest := self.ledger.get_execution_job(job_id)
                        )
                        is not None
                        and latest.state == JobState.CANCELLED
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
                    self._pending_cancellations.discard(job_id)
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
                "CUDA subprocess failed without a verifiable device loss",
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
            "CUDA subprocess failed while its assigned device remained available",
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
        if current.state in (
            JobState.CANCELLED,
            JobState.SUCCEEDED,
            JobState.FAILED,
        ):
            return
        transitioned = self.ledger.request_attempt_cancel(
            job.job_id,
            job.attempt,
            attempt_id=job.attempt_id,
        )
        if transitioned:
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.CANCELLING.value,
            )
        self.ledger.finish_attempt(
            job.job_id,
            job.attempt,
            JobState.CANCELLED,
            attempt_id=job.attempt_id,
            exit_code=exit_code,
        )
        self._record_transition(
            job,
            JobState.CANCELLING.value,
            JobState.CANCELLED.value,
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
        if current.state in (
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
        ):
            return
        if current.state == JobState.CANCELLING:
            self._finish_cancelled(job, failure.exit_code)
            return
        try:
            self.ledger.finish_attempt(
                job.job_id,
                job.attempt,
                JobState.FAILED,
                attempt_id=job.attempt_id,
                error_code=failure.code,
                error_message=failure.message,
                exit_code=failure.exit_code,
            )
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.FAILED.value,
                code=failure.code.value,
            )
        except (ServiceError, ValueError):
            # If cancel committed its PostgreSQL transition first, the attempted
            # RUNNING -> FAILED transition observes CANCELLING. Re-read and
            # complete cancellation; otherwise preserve the original error.
            latest = self.ledger.get_execution_job(job.job_id)
            if (
                latest is not None
                and self._same_attempt(latest, job)
                and latest.state == JobState.CANCELLING
            ):
                self._finish_cancelled(job, failure.exit_code)
                return
            if latest is not None and latest.state in (
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
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
        if current.state in (
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
        ):
            return False
        if current.state == JobState.CANCELLING:
            self._finish_cancelled(job, failure.exit_code)
            return False
        try:
            self.ledger.schedule_retry(
                job.job_id,
                job.attempt,
                attempt_id=job.attempt_id,
                error_code=failure.code,
                error_message=failure.message,
                exit_code=failure.exit_code,
            )
        except (ServiceError, ValueError):
            latest = self.ledger.get_execution_job(job.job_id)
            if (
                latest is not None
                and self._same_attempt(latest, job)
                and latest.state == JobState.CANCELLING
            ):
                self._finish_cancelled(job, failure.exit_code)
                return False
            if latest is not None and latest.state in (
                JobState.RETRYING,
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
            ):
                return (
                    self._same_attempt(latest, job)
                    and latest.state == JobState.RETRYING
                )
            raise
        self._record_transition(
            job,
            JobState.RUNNING.value,
            JobState.RETRYING.value,
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
