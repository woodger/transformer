from app.flight.constants import JobState, STATE_TRANSITIONS, TERMINAL_STATES


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
