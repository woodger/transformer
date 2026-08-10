import threading
import uuid
from dataclasses import replace
from types import SimpleNamespace

from app.flight.constants import ErrorCode
from app.flight.records import ExecutionJobRecord
from app.flight.worker_attempt import WorkerAttemptExecutor
from app.flight.worker_subprocess import WorkerSubprocessError
from app.service.domain.job import ExecutionState, InputState


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
                execution_state=ExecutionState.RETRYING,
            )

    def mark_second_attempt_running(self):
        with self._lock:
            self.current = replace(
                self.current,
                execution_state=ExecutionState.RUNNING,
                attempt=2,
                attempt_id=str(uuid.uuid4()),
                assigned_device_id="GPU-b",
            )
            return self.current

    def mark_cancelling(self):
        with self._lock:
            self.current = replace(
                self.current,
                execution_state=ExecutionState.CANCELLING,
            )

    def finish_attempt(self, _job_id, attempt, target_state, **_fields):
        with self._lock:
            assert self.current.attempt == attempt
            self.current = replace(
                self.current,
                execution_state=ExecutionState(target_state),
            )

    def request_attempt_cancel(self, _job_id, attempt, *, attempt_id):
        with self._lock:
            assert self.current.attempt == attempt
            assert self.current.attempt_id == attempt_id
            if self.current.execution_state == ExecutionState.CANCELLING:
                return False
            assert self.current.execution_state == ExecutionState.RUNNING
            self.current = replace(
                self.current,
                execution_state=ExecutionState.CANCELLING,
            )
            return True


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
    def __init__(self):
        self.cleaned = []

    def cleanup_unpublished(self, _job):
        self.cleaned.append((_job.attempt, _job.attempt_id))


def _job() -> ExecutionJobRecord:
    return ExecutionJobRecord(
        job_id="11111111-1111-4111-8111-111111111111",
        owner_subject="inventory",
        operation="fit",
        input_state=InputState.OPEN,
        execution_state=ExecutionState.RUNNING,
        input_revision=1,
        selected_device="cuda",
        model_label="daily",
        input_model_ref=None,
        prediction_column="out",
        model_config=None,
        training_config=None,
        data_contract={"data_contract_sha256": "d" * 64},
        config_hash="a" * 64,
        manifest_sha256=None,
        feature_dim=2,
        input_frame_count=1,
        attempt=1,
        assigned_device_id="GPU-a",
        resume_generation=None,
        queued_at=1.0,
        started_at=2.0,
        attempt_id=str(uuid.uuid4()),
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
        ledger,
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
        resumable_fit=True,
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
        assert (
            ledger.get_execution_job(first.job_id).execution_state
            == ExecutionState.CANCELLED
        )
    finally:
        executor.interrupt_for_shutdown()
        first_thread.join(2)
        if second_thread is not None:
            second_thread.join(2)


def test_stale_executor_cannot_mutate_a_new_attempt():
    first = _job()
    ledger = _Ledger(first)
    second = ledger.mark_second_attempt_running()
    artifacts = _Artifacts()

    class UnexpectedRunner:
        def run(self, *_args, **_kwargs):
            raise AssertionError("stale attempt started a subprocess")

    executor = WorkerAttemptExecutor(
        ledger,
        SimpleNamespace(
            build=lambda *_args, **_kwargs: SimpleNamespace(inputs=())
        ),
        UnexpectedRunner(),
        artifacts,
        logger=_Logger(),
        metrics=_Metrics(),
    )

    executor.execute(first)

    assert ledger.get_execution_job(first.job_id) == second
    assert artifacts.cleaned == [(first.attempt, first.attempt_id)]
