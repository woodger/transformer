from __future__ import annotations

from dataclasses import dataclass, field
import errno
import hashlib
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Callable, Sequence

from app.config import PROJECT_ROOT
from app.flight.constants import ErrorCode, JobState
from app.flight.contract import model_config_to_api, train_config_to_api
from app.flight.errors import ServiceError
from app.flight.observability import JsonLogger, OperationalMetrics
from app.flight.worker_plan import (
    ExecutionInput,
    WorkerPlanBuilder,
    WorkerPlanError,
)
from app.flight.worker_subprocess import (
    WorkerSubprocessError,
    WorkerSubprocessRunner,
)
from app.storage.checkpoint import CHECKPOINT_FORMAT, load_checkpoint_metadata
from app.training.run_config import ModelConfig, TrainConfig


_COPY_CHUNK_BYTES = 1024 * 1024
_DISK_FULL_ERRNOS = {
    value
    for value in (errno.ENOSPC, getattr(errno, "EDQUOT", None))
    if value is not None
}


@dataclass(frozen=True)
class _Failure(Exception):
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


class WorkerPool:
    """Durable single-instance worker pool for queued Flight jobs.

    PostgreSQL records lifecycle transitions; in-process FIFO queues wake the
    CPU and CUDA lanes without polling the database. Each claimed job gets
    exactly one legacy CLI subprocess.
    """

    def __init__(
        self,
        config,
        ledger,
        spool,
        *,
        logger: JsonLogger | None = None,
        metrics: OperationalMetrics | None = None,
        popen_factory: Callable = subprocess.Popen,
        argv_hook: Callable[[dict, tuple[str, ...]], Sequence[str]] | None = None,
        signal_group: Callable[[int, int], None] = os.killpg,
        python_executable: str | None = None,
        cli_path: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if config.cpu_capacity <= 0:
            raise ValueError("cpu_capacity must be greater than zero")
        if config.cuda_capacity != 1:
            raise ValueError("Flight contract v1 requires exactly one CUDA lane")
        self.config = config
        self.ledger = ledger
        self.spool = spool
        self.logger = logger or JsonLogger()
        self.metrics = metrics or OperationalMetrics()
        self._popen = popen_factory
        self._argv_hook = argv_hook
        self._signal_group = signal_group
        self._python = python_executable or sys.executable
        self._cli_path = cli_path or os.path.join(PROJECT_ROOT, "app", "main.py")
        self._monotonic = monotonic
        self._plan_builder = WorkerPlanBuilder(
            config,
            ledger,
            spool,
            python_executable=self._python,
            cli_path=self._cli_path,
        )
        self._subprocess_runner = WorkerSubprocessRunner(
            config,
            ledger,
            spool,
            logger=self.logger,
            popen_factory=self._popen,
            signal_group=self._signal_group,
            python_executable=self._python,
            monotonic=self._monotonic,
        )

        self._lifecycle_lock = threading.Lock()
        self._claim_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active: dict[str, _ActiveAttempt] = {}
        self._pending_cancellations: set[str] = set()
        self._threads: list[threading.Thread] = []
        self._queues: dict[str, queue.Queue[str | None]] = {
            "cpu": queue.Queue(),
            "cuda": queue.Queue(),
        }
        self._queue_lock = threading.Lock()
        self._enqueued: set[str] = set()
        self._stop_claiming = threading.Event()
        self._force_stop = threading.Event()
        self._started = False

    @property
    def lane_counts(self) -> dict[str, int]:
        return {"cpu": self.config.cpu_capacity, "cuda": 1}

    def start(self) -> "WorkerPool":
        with self._lifecycle_lock:
            if self._started:
                return self
            self._started = True
            for job in self.ledger.queued_jobs():
                self._enqueue(job)
            for device, count in self.lane_counts.items():
                for index in range(count):
                    worker_id = f"{os.getpid()}-{device}-{index}"
                    thread = threading.Thread(
                        target=self._lane,
                        args=(device, worker_id),
                        name=f"transformer-flight-{device}-{index}",
                        daemon=True,
                    )
                    self._threads.append(thread)
                    thread.start()
        return self

    def notify_queued(self, job_id: str | None = None) -> None:
        """Enqueue a job after the coordinator commits QUEUED."""
        if job_id is None:
            for job in self.ledger.queued_jobs():
                self._enqueue(job)
            return
        job = self.ledger.get_job(job_id)
        if job is not None:
            self._enqueue(job)

    def notify_cancel(self, job_id: str) -> None:
        """Cancellation callback used after RUNNING -> CANCELLING commits."""
        with self._active_lock:
            active = self._active.get(job_id)
            if active is None:
                self._pending_cancellations.add(job_id)
            else:
                active.cancel.set()

    def stop_claiming(self) -> None:
        """Close the durable queue-claim boundary without cancelling work."""
        with self._claim_lock:
            if self._stop_claiming.is_set():
                return
            self._stop_claiming.set()
            for device, count in self.lane_counts.items():
                for _ in range(count):
                    self._queues[device].put(None)

    def shutdown(self, timeout: float | None = None) -> None:
        self.stop_claiming()
        drain = self.config.shutdown_drain_seconds if timeout is None else max(0.0, timeout)
        deadline = self._monotonic() + drain
        for thread in tuple(self._threads):
            remaining = max(0.0, deadline - self._monotonic())
            thread.join(remaining)
        if any(thread.is_alive() for thread in self._threads):
            self._force_stop.set()
            with self._active_lock:
                for active in self._active.values():
                    active.cancel.set()
            for thread in tuple(self._threads):
                thread.join(self.config.cancel_grace_seconds + 1.0)
        if any(thread.is_alive() for thread in self._threads):
            self.logger.event(
                "flight.worker.drain_exceeded",
                activeWorkers=sum(thread.is_alive() for thread in self._threads),
            )
            # The process-level state lock cannot be released while a worker
            # thread can still mutate PostgreSQL or the spool. Process groups have
            # already received SIGKILL; finish joining before returning control
            # to FlightApplication.shutdown().
            for thread in tuple(self._threads):
                thread.join()

    def run_once(self, device: str, *, worker_id: str | None = None) -> bool:
        """Atomically claim and execute at most one job for deterministic tests."""
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self._stop_claiming.is_set():
            return False
        job_id = self._dequeue(device)
        if job_id is None:
            for job in self.ledger.queued_jobs():
                self._enqueue(job)
            job_id = self._dequeue(device)
        if job_id is None:
            return False
        return self._claim_and_execute(
            device,
            job_id,
            worker_id or f"{os.getpid()}-{device}-manual",
        )

    def _claim_and_execute(self, device: str, job_id: str, worker_id: str) -> bool:
        with self._claim_lock:
            if self._stop_claiming.is_set():
                return False
            claimed = self.ledger.claim_job(
                job_id,
                device,
                worker_id=worker_id,
            )
        if claimed is None:
            return False
        queue_wait = max(
            0.0,
            float(claimed["started_at"]) - float(claimed["queued_at"]),
        )
        self.metrics.add("jobsStarted")
        self.metrics.add("workerQueueWaitSeconds", queue_wait)
        self.metrics.record_transition(
            JobState.QUEUED.value,
            JobState.RUNNING.value,
        )
        self.logger.event(
            "flight.job.transition",
            jobId=claimed["job_id"],
            attempt=claimed["attempt"],
            device=device,
            fromState=JobState.QUEUED.value,
            toState=JobState.RUNNING.value,
            queueWaitSeconds=queue_wait,
        )
        self._execute_claimed(claimed)
        return True

    def build_argv(self, job: dict, attempt: int | None = None) -> list[str]:
        """Render the trusted CLI invocation; exposed for contract-level tests."""
        attempt = job["attempt"] if attempt is None else attempt
        try:
            return list(self._plan_builder.build_argv(
                job,
                attempt,
                argv_hook=self._argv_hook,
            ))
        except WorkerPlanError as exc:
            raise _Failure(exc.code, exc.message) from exc

    def _lane(self, device: str, worker_id: str) -> None:
        while True:
            job_id = self._queues[device].get()
            if job_id is None:
                return
            with self._queue_lock:
                self._enqueued.discard(job_id)
            try:
                self._claim_and_execute(device, job_id, worker_id)
            except Exception as exc:
                # A single unexpected attempt-finalization/storage error must
                # not permanently remove capacity from the service.  The
                # claimed job is never requeued here (especially not fit); its
                # durable state is left for explicit recovery/reconciliation.
                self.metrics.add("workerLaneErrors")
                self.logger.event(
                    "flight.worker.lane_error",
                    device=device,
                    workerId=worker_id,
                    errorType=type(exc).__name__,
                )

    def _enqueue(self, job: dict) -> None:
        if job.get("state") != JobState.QUEUED.value:
            return
        device = job.get("selected_device")
        if device not in self._queues:
            return
        job_id = job["job_id"]
        with self._queue_lock:
            if job_id in self._enqueued or self._stop_claiming.is_set():
                return
            self._enqueued.add(job_id)
            self._queues[device].put(job_id)

    def _dequeue(self, device: str) -> str | None:
        try:
            job_id = self._queues[device].get_nowait()
        except queue.Empty:
            return None
        if job_id is None:
            return None
        with self._queue_lock:
            self._enqueued.discard(job_id)
        return job_id

    def _execute_claimed(self, job: dict) -> None:
        job_id = job["job_id"]
        attempt = job["attempt"]
        active = _ActiveAttempt(job_id, attempt)
        with self._active_lock:
            self._active[job_id] = active
            if job_id in self._pending_cancellations:
                self._pending_cancellations.remove(job_id)
                active.cancel.set()

        started = self._monotonic()
        succeeded = False
        try:
            current = self.ledger.get_job(job_id)
            if current is None:
                raise _Failure(ErrorCode.INTERNAL, "claimed job disappeared")
            if current["state"] == JobState.CANCELLING.value:
                active.cancel.set()
                raise _Failure(ErrorCode.CANCELLED, "job cancellation was requested")

            self.spool.ensure_free_space(self.config.disk_min_free_bytes)
            try:
                plan = self._plan_builder.build(
                    job,
                    attempt,
                    argv_hook=self._argv_hook,
                )
            except WorkerPlanError as exc:
                raise _Failure(exc.code, exc.message) from exc
            try:
                result = self._subprocess_runner.run(
                    job,
                    plan,
                    cancel=active.cancel,
                    force_stop=self._force_stop,
                )
            except WorkerSubprocessError as exc:
                raise _Failure(
                    exc.code,
                    exc.message,
                    exc.exit_code,
                ) from exc
            if job["operation"] == "predict":
                self._publish_outputs(job, plan.inputs, result.outputs)
            else:
                self._publish_model(job)
            succeeded = True
            self.metrics.add("jobsSucceeded")
        except BaseException as exc:
            current = self.ledger.get_job(job_id)
            if current is not None and current["state"] == JobState.SUCCEEDED.value:
                succeeded = True
            else:
                self._cleanup_unpublished(job)
                failure = self._coerce_failure(exc)
                if failure.code == ErrorCode.CUDA_OUT_OF_MEMORY:
                    self.metrics.add("cudaOutOfMemory")
                    self.logger.event(
                        "flight.cuda.oom",
                        jobId=job_id,
                        attempt=attempt,
                        device=job["selected_device"],
                    )
                elif failure.code == ErrorCode.DEVICE_UNAVAILABLE:
                    self.metrics.add("cudaUnavailableDuringExecution")
                # Cleanup can race with a committed cancel action.  Refresh
                # state before choosing the terminal outcome so cancel wins
                # whenever it was registered before final artifact commit.
                current = self.ledger.get_job(job_id)
                if (
                    current is not None
                    and current["state"] == JobState.CANCELLING.value
                ) or failure.code == ErrorCode.CANCELLED:
                    self._finish_cancelled(job, failure.exit_code)
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
                self._active.pop(job_id, None)
                self._pending_cancellations.discard(job_id)
            elapsed = self._monotonic() - started
            self.metrics.add("workerRunSeconds", elapsed)
            self.logger.event(
                "flight.worker.finished",
                jobId=job_id,
                attempt=attempt,
                device=job["selected_device"],
                succeeded=succeeded,
                runSeconds=elapsed,
            )

    def _publish_outputs(
        self,
        job: dict,
        inputs: tuple[ExecutionInput, ...],
        outputs: tuple[dict, ...],
    ) -> None:
        if len(outputs) != len(inputs):
            raise _Failure(
                ErrorCode.MALFORMED_OUTPUT,
                "prediction subprocess output count did not match input count",
            )
        result = {
            "outputs": [
                {"ordinal": item["ordinal"], "rows": item["rows"]}
                for item in outputs
            ]
        }
        self.ledger.publish_outputs(
            job["job_id"],
            job["attempt"],
            outputs,
            result=result,
        )
        for item in outputs:
            self.metrics.add("predictionOutputBytes", item["bytes"])
            self.metrics.add("predictionOutputRows", item["rows"])
            self.metrics.add("predictionOutputBatches", item["batches"])
            self.logger.event(
                "flight.output.published",
                jobId=job["job_id"],
                ordinal=item["ordinal"],
                rows=item["rows"],
                batches=item["batches"],
                bytes=item["bytes"],
            )
        self._record_transition(
            job,
            JobState.RUNNING.value,
            JobState.SUCCEEDED.value,
        )

    def _publish_model(self, job: dict) -> None:
        attempt_path = self.spool.attempt_checkpoint_path(
            job["job_id"], job["attempt"]
        )
        if not os.path.isfile(attempt_path):
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess did not create a checkpoint",
            )
        try:
            checkpoint = load_checkpoint_metadata(attempt_path, "cpu")
            actual_model = ModelConfig.from_dict(checkpoint.get("model_config"))
            actual_train = TrainConfig.from_dict(checkpoint.get("train_config"))
        except Exception as exc:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an invalid checkpoint",
            ) from exc
        expected_model = ModelConfig.from_dict(job["model_config"])
        expected_train = TrainConfig.from_dict(job["training_config"])
        if checkpoint.get("format") != CHECKPOINT_FORMAT or actual_model is None:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an unsupported checkpoint",
            )
        expected_values = expected_model.to_dict()
        expected_values["feature_dim"] = job.get("feature_dim")
        if actual_model.to_dict() != expected_values or actual_train != expected_train:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit checkpoint configuration differs from the immutable job config",
            )

        digest = _sha256_file(attempt_path)
        byte_count = os.path.getsize(attempt_path)
        model_ref = f"mdl_{uuid.uuid4().hex}"
        model_directory = self.spool.model_directory(model_ref)
        checkpoint_path = self.spool.model_checkpoint_path(model_ref)
        metadata_path = self.spool.model_metadata_path(model_ref)
        try:
            safe_checkpoint = {
                "format": checkpoint["format"],
                "serviceVersion": checkpoint.get("version"),
                "sha256": digest,
                "bytes": byte_count,
                "modelConfig": model_config_to_api(actual_model),
                "trainConfig": train_config_to_api(actual_train),
                "dataSchema": _data_schema_to_api(
                    checkpoint.get("data_schema"),
                    actual_model,
                ),
                "checkpointSelection": _checkpoint_selection_to_api(
                    (checkpoint.get("extra") or {}).get("checkpoint_selection"),
                    actual_train,
                ),
            }
        except _Failure:
            raise
        except Exception as exc:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit subprocess created an invalid checkpoint",
            ) from exc
        if not isinstance(safe_checkpoint["serviceVersion"], str) or not safe_checkpoint[
            "serviceVersion"
        ]:
            raise _Failure(
                ErrorCode.SUBPROCESS_FAILED,
                "fit checkpoint does not contain a service version",
            )
        metadata = {
            "modelRef": model_ref,
            "label": job["model_label"],
            # Internal snake_case copies let predict create resolve a generation
            # without depending on the public status document representation.
            "model_config": actual_model.to_dict(),
            "train_config": actual_train.to_dict(),
            "checkpoint": safe_checkpoint,
        }
        try:
            with open(attempt_path, "rb") as source:
                with self.spool.staged_file(checkpoint_path) as (target, _):
                    shutil.copyfileobj(source, target, _COPY_CHUNK_BYTES)
            self.spool.atomic_write_json(metadata_path, metadata)
            self.ledger.publish_model(
                job["job_id"],
                job["attempt"],
                model_ref=model_ref,
                label=job["model_label"],
                generation=None,
                checkpoint_path=self.spool.model_relative_path(checkpoint_path),
                metadata_path=self.spool.model_relative_path(metadata_path),
                sha256=digest,
                metadata=metadata,
                result={"modelRef": model_ref, "checkpoint": safe_checkpoint},
            )
            self.metrics.add("checkpointBytes", byte_count)
            self.logger.event(
                "flight.model.published",
                jobId=job["job_id"],
                modelRef=model_ref,
                bytes=byte_count,
                sha256=digest,
            )
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.SUCCEEDED.value,
            )
        except BaseException:
            current = self.ledger.get_job(job["job_id"])
            if current is None or current["state"] != JobState.SUCCEEDED.value:
                self.spool.remove(model_directory)
            raise

    def _cleanup_unpublished(self, job: dict) -> None:
        try:
            inputs = self.ledger.list_inputs(job["job_id"])
        except Exception as exc:
            self.logger.event(
                "flight.worker.cleanup_failed",
                jobId=job["job_id"],
                attempt=job["attempt"],
                artifact="attempt",
                errorType=type(exc).__name__,
            )
            return
        paths = [
            self.spool.attempt_output_path(
                job["job_id"], job["attempt"], item["ordinal"]
            )
            for item in inputs
        ]
        paths.append(
            self.spool.attempt_checkpoint_path(job["job_id"], job["attempt"])
        )
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                self.spool.remove(path)
            except Exception as exc:
                # Publication is ledger-gated.  A filesystem cleanup failure
                # can leave only an invisible orphan, which startup
                # reconciliation removes; it must not strand the job RUNNING
                # or kill a worker lane.
                self.logger.event(
                    "flight.worker.cleanup_failed",
                    jobId=job["job_id"],
                    attempt=job["attempt"],
                    artifact="attempt",
                    errorType=type(exc).__name__,
                )

    def _finish_cancelled(self, job: dict, exit_code: int | None) -> None:
        current = self.ledger.get_job(job["job_id"])
        if current is None or current["state"] in (
            JobState.CANCELLED.value,
            JobState.SUCCEEDED.value,
            JobState.FAILED.value,
        ):
            return
        if current["state"] == JobState.RUNNING.value:
            self.ledger.transition_job(job["job_id"], JobState.CANCELLING)
            self._record_transition(
                job,
                JobState.RUNNING.value,
                JobState.CANCELLING.value,
            )
        self.ledger.finish_attempt(
            job["job_id"],
            job["attempt"],
            JobState.CANCELLED,
            exit_code=exit_code,
        )
        self._record_transition(
            job,
            JobState.CANCELLING.value,
            JobState.CANCELLED.value,
        )

    def _finish_failed(self, job: dict, failure: _Failure) -> None:
        current = self.ledger.get_job(job["job_id"])
        if current is None or current["state"] in (
            JobState.SUCCEEDED.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
        ):
            return
        if current["state"] == JobState.CANCELLING.value:
            self._finish_cancelled(job, failure.exit_code)
            return
        try:
            self.ledger.finish_attempt(
                job["job_id"],
                job["attempt"],
                JobState.FAILED,
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
            # RUNNING -> FAILED transition observes CANCELLING.  Re-read and
            # complete cancellation; otherwise preserve the original error.
            latest = self.ledger.get_job(job["job_id"])
            if latest is not None and latest["state"] == JobState.CANCELLING.value:
                self._finish_cancelled(job, failure.exit_code)
                return
            if latest is not None and latest["state"] in (
                JobState.SUCCEEDED.value,
                JobState.FAILED.value,
                JobState.CANCELLED.value,
            ):
                return
            raise

    def _record_transition(
        self,
        job: dict,
        from_state: str,
        to_state: str,
        *,
        code: str | None = None,
    ) -> None:
        self.metrics.record_transition(from_state, to_state)
        fields = {
            "jobId": job["job_id"],
            "attempt": job["attempt"],
            "device": job["selected_device"],
            "fromState": from_state,
            "toState": to_state,
        }
        if code is not None:
            fields["code"] = code
        self.logger.event("flight.job.transition", **fields)

    @staticmethod
    def _coerce_failure(exc: BaseException) -> _Failure:
        if isinstance(exc, _Failure):
            return exc
        if isinstance(exc, ServiceError):
            return _Failure(exc.code, exc.message)
        if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
            return _Failure(ErrorCode.DISK_FULL, "runtime directory is full")
        return _Failure(ErrorCode.INTERNAL, "worker execution failed internally")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_schema_to_api(value: dict | None, model_config: ModelConfig) -> dict:
    if not isinstance(value, dict):
        raise _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint does not contain data schema metadata",
        )
    source = value.get("src")
    target = value.get("tgt")
    missing = value.get("missing")
    if not all(isinstance(item, dict) for item in (source, target, missing)):
        raise _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid data schema metadata",
        )
    result = {
        "schemaVersion": value.get("schema_version"),
        "tensorDtype": value.get("tensor_dtype"),
        "source": {
            "column": source.get("column"),
            "acceptedElementTypes": source.get("accepted_element_types"),
            "width": source.get("width"),
        },
        "target": {
            "column": target.get("column"),
            "acceptedElementTypes": target.get("accepted_element_types"),
            "width": target.get("width"),
        },
        "featureDim": value.get("feature_dim"),
        "modelInputFeatureDim": value.get("model_input_feature_dim"),
        "contextMode": value.get("context_mode"),
        "normalization": value.get("normalization"),
        "missing": {
            "nanFill": missing.get("nan_fill"),
            "flags": missing.get("flags"),
        },
    }
    expected_types = ["float32", "float64"]
    expected_input_dim = (
        model_config.feature_dim * 2
        if model_config.context_mode == "relaxed"
        else model_config.feature_dim
    )
    if (
        result["schemaVersion"] != 1
        or result["tensorDtype"] != "float32"
        or result["source"]["column"] != "src"
        or result["source"]["acceptedElementTypes"] != expected_types
        or type(result["source"]["width"]) is not int
        or result["source"]["width"] <= 0
        or result["target"]["column"] != "tgt"
        or result["target"]["acceptedElementTypes"] != expected_types
        or result["target"]["width"] != 6
        or result["featureDim"] != model_config.feature_dim
        or result["source"]["width"]
        != model_config.seq_len * model_config.feature_dim
        or result["modelInputFeatureDim"] != expected_input_dim
        or result["contextMode"] != model_config.context_mode
        or result["normalization"] is not None
        or result["missing"]["nanFill"] != 0.0
        or result["missing"]["flags"]
        != ("per-feature" if model_config.context_mode == "relaxed" else "none")
    ):
        raise _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid data schema metadata",
        )
    return result


def _checkpoint_selection_to_api(
    value: dict | None,
    train_config: TrainConfig,
) -> dict:
    value = value if isinstance(value, dict) else {}
    result = {
        "monitor": value.get("monitor", train_config.monitor),
        "monitorMinImprovement": value.get(
            "monitor_min_improvement",
            train_config.monitor_min_improvement,
        ),
        "bestMonitor": value.get("best_monitor"),
        "bestFrame": value.get("best_frame"),
        "bestEpoch": value.get("best_epoch"),
        "baselinePassed": bool(value.get("baseline_passed", False)),
        "source": value.get("source", "current"),
    }
    best_monitor = result["bestMonitor"]
    best_frame = result["bestFrame"]
    best_epoch = result["bestEpoch"]
    if (
        result["monitor"] != train_config.monitor
        or result["monitorMinImprovement"] != train_config.monitor_min_improvement
        or (
            best_monitor is not None
            and (
                isinstance(best_monitor, bool)
                or not isinstance(best_monitor, (int, float))
                or not math.isfinite(best_monitor)
            )
        )
        or (
            best_frame is not None
            and (type(best_frame) is not int or best_frame < 0)
        )
        or (
            best_epoch is not None
            and (type(best_epoch) is not int or best_epoch < 0)
        )
        or (
            "baseline_passed" in value
            and not isinstance(value["baseline_passed"], bool)
        )
        or result["source"] not in ("best_monitor", "current")
    ):
        raise _Failure(
            ErrorCode.SUBPROCESS_FAILED,
            "fit checkpoint contains invalid checkpoint selection metadata",
        )
    return result
