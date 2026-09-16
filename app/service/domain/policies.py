from dataclasses import dataclass

from app.service.domain.job import (
    EXECUTION_STATE_TRANSITIONS,
    IMMEDIATE_CANCEL_STATES,
    INPUT_STATE_TRANSITIONS,
    TERMINAL_EXECUTION_STATES,
    ErrorCode,
    ExecutionState,
    InputState,
)


@dataclass(frozen=True, slots=True)
class CancelDecision:
    target: ExecutionState | None
    notify_worker: bool
    abort_open_input: bool


@dataclass(frozen=True, slots=True)
class StartupDecision:
    target: ExecutionState
    error_code: ErrorCode | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class AttemptOutcomeDecision:
    target: ExecutionState


@dataclass(frozen=True, slots=True)
class DeviceDecision:
    selected: str | None
    error_code: ErrorCode | None


def validate_execution_transition(
    current: str | ExecutionState,
    target: str | ExecutionState,
) -> None:
    current_state = ExecutionState(current)
    target_state = ExecutionState(target)
    if target_state not in EXECUTION_STATE_TRANSITIONS[current_state]:
        raise ValueError(
            f"invalid execution state transition: {current_state.value} -> "
            f"{target_state.value}"
        )


def validate_input_transition(
    current: str | InputState,
    target: str | InputState,
) -> None:
    current_state = InputState(current)
    target_state = InputState(target)
    if target_state not in INPUT_STATE_TRANSITIONS[current_state]:
        raise ValueError(
            f"invalid input state transition: {current_state.value} -> "
            f"{target_state.value}"
        )


def is_terminal(state: str | ExecutionState) -> bool:
    return ExecutionState(state) in TERMINAL_EXECUTION_STATES


def decide_cancel(state: str | ExecutionState) -> CancelDecision:
    current = ExecutionState(state)
    if current in IMMEDIATE_CANCEL_STATES:
        return CancelDecision(
            target=ExecutionState.CANCELLED,
            notify_worker=False,
            abort_open_input=True,
        )
    if current == ExecutionState.RUNNING:
        return CancelDecision(
            target=ExecutionState.CANCELLING,
            notify_worker=True,
            abort_open_input=True,
        )
    return CancelDecision(
        target=None,
        notify_worker=False,
        abort_open_input=False,
    )


def decide_startup_interruption(
    state: str | ExecutionState,
) -> StartupDecision:
    current = ExecutionState(state)
    if current in (
        ExecutionState.WAITING_INPUT,
        ExecutionState.QUEUED,
        ExecutionState.RUNNING,
        ExecutionState.RETRYING,
    ):
        return StartupDecision(
            target=ExecutionState.FAILED,
            error_code=ErrorCode.EXECUTION_INTERRUPTED,
            error_message="job was interrupted by service restart",
        )
    if current == ExecutionState.CANCELLING:
        return StartupDecision(
            target=ExecutionState.CANCELLED,
            error_code=None,
            error_message=None,
        )
    raise ValueError(
        f"job state cannot be reconciled at startup: {current.value}"
    )


def decide_attempt_outcome(
    state: str | ExecutionState,
    failure: str | ErrorCode,
    *,
    operation: str,
    resumable_fit: bool,
) -> AttemptOutcomeDecision:
    current = ExecutionState(state)
    error = ErrorCode(failure)
    if current == ExecutionState.CANCELLING or error == ErrorCode.CANCELLED:
        return AttemptOutcomeDecision(ExecutionState.CANCELLED)
    retryable_device_loss = error == ErrorCode.DEVICE_LOST and (
        operation == "predict" or (operation == "fit" and resumable_fit)
    )
    retryable_fit_process = (
        error in {
            ErrorCode.EXECUTION_INTERRUPTED,
            ErrorCode.SUBPROCESS_FAILED,
            ErrorCode.SUBPROCESS_HUNG,
        }
        and operation == "fit"
        and resumable_fit
    )
    if retryable_device_loss or retryable_fit_process:
        return AttemptOutcomeDecision(ExecutionState.RETRYING)
    return AttemptOutcomeDecision(ExecutionState.FAILED)


def resolve_device(requested: str, cuda_available: bool) -> DeviceDecision:
    if requested == "cuda":
        return DeviceDecision(
            selected="cuda" if cuda_available else None,
            error_code=(None if cuda_available else ErrorCode.DEVICE_UNAVAILABLE),
        )
    if requested == "auto":
        return DeviceDecision(
            selected="cuda" if cuda_available else "cpu",
            error_code=None,
        )
    if requested == "cpu":
        return DeviceDecision(selected="cpu", error_code=None)
    raise ValueError(f"unsupported requested device: {requested}")
