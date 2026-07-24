from dataclasses import dataclass

from app.flight.constants import (
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
