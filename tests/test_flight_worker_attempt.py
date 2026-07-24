import threading
from dataclasses import replace
from types import SimpleNamespace

from app.flight.constants import ErrorCode, JobState
from app.flight.records import ExecutionJobRecord
from app.flight.worker_attempt import WorkerAttemptExecutor
from app.flight.worker_subprocess import WorkerSubprocessError


class _Ledger:
    def __init__(self, job):
        self._lock = threading.Lock()
        self.current = job

    def get_execution_job(self, _job_id):
        with self._lock:
            return self.current

    def schedule_retry(self, _job_id, attempt, **_fields):
        with self._lock:
            assert self.current.attempt == attempt
            self.current = replace(
                self.current,
                state=JobState.RETRYING,
            )

    def mark_second_attempt_running(self):
        with self._lock:
            self.current = replace(
                self.current,
                state=JobState.RUNNING,
                attempt=2,
                assigned_device_id="GPU-b",
            )
            return self.current

    def mark_cancelling(self):
        with self._lock:
            self.current = replace(
                self.current,
                state=JobState.CANCELLING,
            )

    def finish_attempt(self, _job_id, attempt, target_state, **_fields):
        with self._lock:
            assert self.current.attempt == attempt
            self.current = replace(
                self.current,
                state=JobState(target_state),
            )


class _Runner:
    def __init__(self):
        self.second_started = threading.Event()

    def run(self, job, _plan, *, cancel, force_stop):
        if job.attempt == 1:
            raise WorkerSubprocessError(
                ErrorCode.DEVICE_LOST,
                "assigned GPU disappeared",
                1,
            )
        self.second_started.set()
        if not cancel.wait(2):
            raise AssertionError(
                "cancellation was not delivered to the second attempt"
            )
        raise WorkerSubprocessError(
            (
                ErrorCode.EXECUTION_INTERRUPTED
                if force_stop.is_set()
                else ErrorCode.CANCELLED
            ),
            "job cancellation was requested",
        )


class _Metrics:
    def add(self, *_args):
        pass

    def record_transition(self, *_args):
        pass


class _Logger:
    def event(self, *_args, **_fields):
        pass


class _Artifacts:
    def cleanup_unpublished(self, _job):
        pass


def _job() -> ExecutionJobRecord:
    return ExecutionJobRecord(
        job_id="11111111-1111-4111-8111-111111111111",
        owner_subject="inventory",
        operation="predict",
        state=JobState.RUNNING,
        selected_device="cuda",
        model_label=None,
        input_model_ref="mdl_test",
        prediction_column="out",
        model_config=None,
        training_config=None,
        config_hash="a" * 64,
        seal_hash="b" * 64,
        feature_dim=2,
        input_frame_count=1,
        attempt=1,
        assigned_device_id="GPU-a",
        resume_generation=None,
        queued_at=1.0,
        started_at=2.0,
    )


def test_retry_handoff_keeps_second_attempt_registered_for_cancel():
    first = _job()
    ledger = _Ledger(first)
    runner = _Runner()
    second_thread = None
    executor = None

    def retry_notifier(_job_id):
        nonlocal second_thread
        second = ledger.mark_second_attempt_running()
        second_thread = threading.Thread(
            target=executor.execute,
            args=(second,),
        )
        second_thread.start()
        assert runner.second_started.wait(2)

    executor = WorkerAttemptExecutor(
        SimpleNamespace(disk_min_free_bytes=1),
        ledger,
        SimpleNamespace(ensure_free_space=lambda _minimum: None),
        SimpleNamespace(
            build=lambda *_args, **_kwargs: SimpleNamespace(
                inputs=(),
            )
        ),
        runner,
        _Artifacts(),
        logger=_Logger(),
        metrics=_Metrics(),
        retry_notifier=retry_notifier,
        confirm_device_loss=lambda _device_id: True,
    )
    first_thread = threading.Thread(
        target=executor.execute,
        args=(first,),
    )
    first_thread.start()
    try:
        first_thread.join(2)
        assert not first_thread.is_alive()
        ledger.mark_cancelling()
        executor.notify_cancel(first.job_id)
        second_thread.join(2)
        assert not second_thread.is_alive()
        assert ledger.get_execution_job(first.job_id).state == JobState.CANCELLED
    finally:
        executor.interrupt_for_shutdown()
        first_thread.join(2)
        if second_thread is not None:
            second_thread.join(2)
