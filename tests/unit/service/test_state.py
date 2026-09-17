import pytest

from app.service.domain.job import ErrorCode, ExecutionState, InputState
from app.service.domain.policies import (
    AttemptOutcomeDecision,
    CancelDecision,
    DeviceDecision,
    StartupDecision,
    decide_attempt_outcome,
    decide_cancel,
    decide_startup_interruption,
    is_terminal,
    resolve_device,
    validate_execution_transition,
    validate_input_transition,
)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (current, target)
        for current in ExecutionState
        for target in ExecutionState
    ],
)
def test_execution_transition_matrix_is_explicit(current, target):
    allowed = {
        ExecutionState.WAITING_INPUT: {
            ExecutionState.QUEUED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.QUEUED: {
            ExecutionState.RUNNING,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.RUNNING: {
            ExecutionState.SUCCEEDED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLING,
            ExecutionState.RETRYING,
        },
        ExecutionState.RETRYING: {
            ExecutionState.RUNNING,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
        },
        ExecutionState.CANCELLING: {ExecutionState.CANCELLED},
        ExecutionState.SUCCEEDED: set(),
        ExecutionState.FAILED: set(),
        ExecutionState.CANCELLED: set(),
    }
    if target in allowed[current]:
        validate_execution_transition(current, target)
    else:
        with pytest.raises(ValueError, match="invalid execution state"):
            validate_execution_transition(current, target)


def test_input_state_is_an_independent_one_way_axis():
    validate_input_transition(InputState.OPEN, InputState.CLOSED)
    validate_input_transition(InputState.OPEN, InputState.ABORTED)
    for current in (InputState.CLOSED, InputState.ABORTED):
        for target in InputState:
            with pytest.raises(ValueError, match="invalid input state"):
                validate_input_transition(current, target)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            ExecutionState.WAITING_INPUT,
            CancelDecision(ExecutionState.CANCELLED, False, True),
        ),
        (
            ExecutionState.QUEUED,
            CancelDecision(ExecutionState.CANCELLED, False, True),
        ),
        (
            ExecutionState.RUNNING,
            CancelDecision(ExecutionState.CANCELLING, True, True),
        ),
        (
            ExecutionState.RETRYING,
            CancelDecision(ExecutionState.CANCELLED, False, True),
        ),
        (
            ExecutionState.CANCELLING,
            CancelDecision(None, False, False),
        ),
        (
            ExecutionState.SUCCEEDED,
            CancelDecision(None, False, False),
        ),
        (
            ExecutionState.FAILED,
            CancelDecision(None, False, False),
        ),
        (
            ExecutionState.CANCELLED,
            CancelDecision(None, False, False),
        ),
    ],
)
def test_cancel_policy_covers_every_execution_state(state, expected):
    assert decide_cancel(state) == expected


@pytest.mark.parametrize(
    "state",
    (
        ExecutionState.WAITING_INPUT,
        ExecutionState.QUEUED,
        ExecutionState.RUNNING,
        ExecutionState.RETRYING,
    ),
)
def test_startup_interruption_terminalizes_every_unfinished_job(state):
    assert decide_startup_interruption(state) == StartupDecision(
        ExecutionState.FAILED,
        ErrorCode.EXECUTION_INTERRUPTED,
        "job was interrupted by service restart",
    )


def test_startup_interruption_preserves_requested_cancellation():
    assert decide_startup_interruption(
        ExecutionState.CANCELLING
    ) == StartupDecision(ExecutionState.CANCELLED, None, None)


@pytest.mark.parametrize(
    ("failure", "operation", "resumable", "target"),
    [
        (ErrorCode.DEVICE_LOST, "predict", False, ExecutionState.RETRYING),
        (ErrorCode.DEVICE_LOST, "fit", True, ExecutionState.RETRYING),
        (ErrorCode.DEVICE_LOST, "fit", False, ExecutionState.FAILED),
        (
            ErrorCode.EXECUTION_INTERRUPTED,
            "fit",
            True,
            ExecutionState.RETRYING,
        ),
        (ErrorCode.SUBPROCESS_FAILED, "fit", True, ExecutionState.RETRYING),
        (ErrorCode.SUBPROCESS_HUNG, "fit", True, ExecutionState.RETRYING),
        (ErrorCode.SUBPROCESS_FAILED, "fit", False, ExecutionState.FAILED),
        (ErrorCode.SUBPROCESS_FAILED, "predict", True, ExecutionState.FAILED),
    ],
)
def test_attempt_outcome_restarts_resumable_fit_from_safe_epoch_boundary(
    failure,
    operation,
    resumable,
    target,
):
    assert decide_attempt_outcome(
        ExecutionState.RUNNING,
        failure,
        operation=operation,
        resumable_fit=resumable,
    ) == AttemptOutcomeDecision(target)


def test_cancel_wins_over_worker_failure():
    assert decide_attempt_outcome(
        ExecutionState.CANCELLING,
        ErrorCode.SUBPROCESS_FAILED,
        operation="fit",
        resumable_fit=True,
    ) == AttemptOutcomeDecision(ExecutionState.CANCELLED)


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        ("cpu", False, DeviceDecision("cpu", None)),
        ("auto", False, DeviceDecision("cpu", None)),
        ("auto", True, DeviceDecision("cuda", None)),
        ("cuda", True, DeviceDecision("cuda", None)),
        ("cuda", False, DeviceDecision(None, ErrorCode.DEVICE_UNAVAILABLE)),
    ],
)
def test_device_resolution_policy(requested, available, expected):
    assert resolve_device(requested, available) == expected


def test_only_execution_terminal_states_are_terminal():
    assert {state for state in ExecutionState if is_terminal(state)} == {
        ExecutionState.SUCCEEDED,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    }
