from dataclasses import dataclass

from app.service.domain.job import (
    IMMEDIATE_CANCEL_STATES,
    STATE_TRANSITIONS,
    TERMINAL_STATES,
    ErrorCode,
    JobState,
)

_EXECUTION_INTERRUPTED_MESSAGE = (
    "worker execution was interrupted by service restart"
)


@dataclass(frozen=True, slots=True)
class CancelDecision:
    target: JobState | None
    notify_worker: bool


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    target: JobState
    error_code: ErrorCode | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class AttemptOutcomeDecision:
    target: JobState


@dataclass(frozen=True, slots=True)
class DeviceDecision:
    selected: str | None
    error_code: ErrorCode | None


def validate_transition(current: str | JobState, target: str | JobState) -> None:
    current_state = JobState(current)
    target_state = JobState(target)
    if target_state not in STATE_TRANSITIONS[current_state]:
        raise ValueError(
            f"invalid job state transition: {current_state.value} -> "
            f"{target_state.value}"
        )


def is_terminal(state: str | JobState) -> bool:
    return JobState(state) in TERMINAL_STATES


def decide_cancel(state: str | JobState) -> CancelDecision:
    current = JobState(state)
    if current in IMMEDIATE_CANCEL_STATES:
        return CancelDecision(
            target=JobState.CANCELLED,
            notify_worker=False,
        )
    if current == JobState.RUNNING:
        return CancelDecision(
            target=JobState.CANCELLING,
            notify_worker=True,
        )
    return CancelDecision(target=None, notify_worker=False)


def decide_interrupted_attempt(
    state: str | JobState,
) -> RecoveryDecision:
    current = JobState(state)
    if current == JobState.RUNNING:
        return RecoveryDecision(
            target=JobState.FAILED,
            error_code=ErrorCode.EXECUTION_INTERRUPTED,
            error_message=_EXECUTION_INTERRUPTED_MESSAGE,
        )
    if current == JobState.CANCELLING:
        return RecoveryDecision(
            target=JobState.CANCELLED,
            error_code=None,
            error_message=None,
        )
    raise ValueError(
        f"job state cannot be reconciled as interrupted: {current.value}"
    )


def decide_attempt_outcome(
    state: str | JobState,
    failure: str | ErrorCode,
    *,
    operation: str,
    resumable_fit: bool,
) -> AttemptOutcomeDecision:
    current = JobState(state)
    error = ErrorCode(failure)
    if current == JobState.CANCELLING or error == ErrorCode.CANCELLED:
        return AttemptOutcomeDecision(JobState.CANCELLED)
    retryable_device_loss = error == ErrorCode.DEVICE_LOST and (
        operation == "predict" or (operation == "fit" and resumable_fit)
    )
    retryable_restart = (
        error == ErrorCode.EXECUTION_INTERRUPTED
        and operation == "fit"
        and resumable_fit
    )
    if retryable_device_loss or retryable_restart:
        return AttemptOutcomeDecision(JobState.RETRYING)
    return AttemptOutcomeDecision(JobState.FAILED)


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
