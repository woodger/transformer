from __future__ import annotations

import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.service.adapters.outbound.postgres.config import DatabaseConfig
from app.service.adapters.outbound.postgres.ledger import Ledger
from app.service.adapters.outbound.postgres.models import Job
from app.service.adapters.outbound.postgres.session import Database
from app.service.application.commands.jobs import CancelJobAction
from app.service.application.messages.jobs import CancelJobCommand
from app.service.application.services.attempt_executor import (
    WorkerAttemptExecutor,
)
from app.service.application.services.errors import AttemptExecutionError
from app.service.domain.errors import ServiceError
from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.records import ExecutionJobRecord

_JOB_ID = "00000000-0000-4000-8000-000000000001"
_EXECUTION_ID = "00000000-0000-4000-8000-000000000002"
_ATTEMPT_ID = "00000000-0000-4000-8000-000000000003"


def _database() -> Database:
    return Database(
        DatabaseConfig(
            host="localhost",
            database="transformer",
            user="transformer",
            password="secret",
        ),
        engine=create_engine("sqlite+pysqlite:///:memory:"),
    )


def _waiting_job(*, fencing_token: int = 1) -> Job:
    return Job(
        job_id=_JOB_ID,
        owner_subject="inventory",
        client_execution_id=_EXECUTION_ID,
        fencing_token=fencing_token,
        execution_state=ExecutionState.RUNNING.value,
        input_state=InputState.OPEN.value,
        manifest_sha256=None,
        input_closed_at=None,
        waiting_for_input=True,
        waiting_input_ordinal=7,
        input_waiting_since=datetime(2026, 9, 17, tzinfo=UTC),
        acquire_grace_until=datetime(2026, 9, 17, tzinfo=UTC),
        revision=4,
    )


def _execution_job() -> ExecutionJobRecord:
    return ExecutionJobRecord(
        job_id=_JOB_ID,
        owner_subject="inventory",
        operation="fit",
        input_state=InputState.OPEN,
        execution_state=ExecutionState.RUNNING,
        input_revision=0,
        selected_device="cuda",
        model_label="test-model",
        input_model_ref=None,
        prediction_column="prediction",
        source_encoding={},
        model_config=None,
        training_config=None,
        data_contract={},
        model_contract={},
        semantic_digests={},
        config_hash="a" * 64,
        manifest_sha256=None,
        feature_dim=1,
        input_frame_count=0,
        attempt=1,
        assigned_device_id=None,
        resume_generation=None,
        queued_at=None,
        started_at=None,
        attempt_id=_ATTEMPT_ID,
    )


def _cancel_command() -> CancelJobCommand:
    return CancelJobCommand(
        owner_subject="inventory",
        request_id="00000000-0000-4000-8000-000000000004",
        idempotency_key="cancel-request",
        request_hash="b" * 64,
        job_id=_JOB_ID,
        client_execution_id=_EXECUTION_ID,
        fencing_token=1,
    )


def test_cancel_stops_running_worker_before_input_upload_lookup():
    job = _waiting_job()
    stopped: list[str] = []

    def uploads(_statement):
        assert stopped == [_JOB_ID]
        assert job.execution_state == ExecutionState.CANCELLING.value
        assert job.input_state == InputState.ABORTED.value
        assert job.waiting_for_input is False
        assert job.waiting_input_ordinal is None
        assert job.input_waiting_since is None
        assert job.acquire_grace_until is None
        return ()

    session = cast(
        Session,
        SimpleNamespace(
            scalar=lambda _statement: job,
            scalars=uploads,
            delete=lambda _record: None,
            flush=lambda: None,
        ),
    )
    database = _database()
    try:
        stored, cleanup = Ledger(database).cancel_job(
            _JOB_ID,
            owner_subject="inventory",
            client_execution_id=_EXECUTION_ID,
            fencing_token=1,
            stop_active_worker=stopped.append,
            now=0.0,
            connection=session,
        )
    finally:
        database.close()

    assert cleanup == ()
    assert stored["execution_state"] == ExecutionState.CANCELLING.value


def test_cancel_does_not_stop_worker_before_fence_validation():
    job = _waiting_job(fencing_token=2)
    stopped: list[str] = []
    session = cast(
        Session,
        SimpleNamespace(
            scalar=lambda _statement: job,
            scalars=lambda _statement: pytest.fail("unexpected upload lookup"),
        ),
    )
    database = _database()
    try:
        with pytest.raises(ServiceError) as captured:
            Ledger(database).cancel_job(
                _JOB_ID,
                owner_subject="inventory",
                client_execution_id=_EXECUTION_ID,
                fencing_token=1,
                stop_active_worker=stopped.append,
                connection=session,
            )
    finally:
        database.close()

    assert captured.value.code is ErrorCode.STALE_FENCE
    assert stopped == []


def test_cancel_notifies_worker_when_lifecycle_persistence_fails():
    command = _cancel_command()
    notifications: list[str] = []
    metrics: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []

    def cancel(_command, *, stop_active_worker):
        stop_active_worker(_JOB_ID)
        raise RuntimeError("injected PostgreSQL failure")

    action = CancelJobAction(
        SimpleNamespace(cancel=cancel),
        cancel_notifier=notifications.append,
        artifact_cleaner=SimpleNamespace(cleanup=lambda _location: None),
        metrics=SimpleNamespace(add=lambda name: metrics.append(name)),
        logger=SimpleNamespace(
            event=lambda event, **fields: events.append((event, fields)),
        ),
    )

    with pytest.raises(RuntimeError, match="injected PostgreSQL failure"):
        action.cancel(command)

    assert notifications == [_JOB_ID]
    assert metrics == ["jobCancellationPersistenceFailures"]
    assert events == [(
        "flight.job.cancel.persistence_failed",
        {
            "requestId": command.request_id,
            "jobId": _JOB_ID,
            "errorType": "RuntimeError",
        },
    )]


def test_explicit_cancel_never_retries_resumable_fit():
    job = _execution_job()
    started = threading.Event()
    failures: list[BaseException] = []
    retries: list[object] = []
    finished: list[ExecutionState] = []

    def run(_job, _plan, *, cancel, force_stop):
        del force_stop
        started.set()
        assert cancel.wait(1.0)
        raise AttemptExecutionError(
            ErrorCode.SUBPROCESS_FAILED,
            "subprocess stopped after cancellation",
            137,
        )

    def finish_attempt(_job_id, _attempt, target_state, **_fields):
        finished.append(target_state)
        return {}

    executor = WorkerAttemptExecutor(
        SimpleNamespace(
            get_execution_job=lambda _job_id: job,
            request_attempt_cancel=lambda *_args, **_fields: True,
            finish_attempt=finish_attempt,
            schedule_retry=lambda *_args, **_fields: retries.append(True),
        ),
        SimpleNamespace(build=lambda _job, _attempt: object()),
        SimpleNamespace(run=run),
        SimpleNamespace(cleanup_unpublished=lambda _job: None),
        logger=SimpleNamespace(event=lambda _event, **_fields: None),
        metrics=SimpleNamespace(
            add=lambda _name, _value=1.0: None,
            record_transition=lambda _from_state, _to_state: None,
        ),
        resumable_fit=True,
    )

    def execute() -> None:
        try:
            executor.execute(job)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=execute)
    thread.start()
    assert started.wait(1.0)

    executor.notify_cancel(_JOB_ID)
    thread.join(1.0)

    assert thread.is_alive() is False
    assert failures == []
    assert retries == []
    assert finished == [ExecutionState.CANCELLED]


def test_pending_explicit_cancel_does_not_launch_worker():
    job = _execution_job()
    launches: list[str] = []
    finished: list[ExecutionState] = []

    def finish_attempt(_job_id, _attempt, target_state, **_fields):
        finished.append(target_state)
        return {}

    executor = WorkerAttemptExecutor(
        SimpleNamespace(
            get_execution_job=lambda _job_id: job,
            request_attempt_cancel=lambda *_args, **_fields: True,
            finish_attempt=finish_attempt,
        ),
        SimpleNamespace(
            build=lambda _job, _attempt: launches.append("plan"),
        ),
        SimpleNamespace(
            run=lambda *_args, **_fields: launches.append("subprocess"),
        ),
        SimpleNamespace(cleanup_unpublished=lambda _job: None),
        logger=SimpleNamespace(event=lambda _event, **_fields: None),
        metrics=SimpleNamespace(
            add=lambda _name, _value=1.0: None,
            record_transition=lambda _from_state, _to_state: None,
        ),
        resumable_fit=True,
    )

    executor.notify_cancel(_JOB_ID)
    executor.execute(job)

    assert launches == []
    assert finished == [ExecutionState.CANCELLED]
