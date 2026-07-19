import pytest

from app.flight.constants import JobState
from app.flight.state import is_terminal, validate_transition


def test_state_machine_accepts_normative_flow_and_immediate_cancel():
    validate_transition(JobState.UPLOADING, JobState.SEALED)
    validate_transition(JobState.SEALED, JobState.QUEUED)
    validate_transition(JobState.QUEUED, JobState.RUNNING)
    validate_transition(JobState.RUNNING, JobState.SUCCEEDED)
    validate_transition(JobState.UPLOADING, JobState.CANCELLED)
    validate_transition(JobState.RUNNING, JobState.CANCELLING)
    validate_transition(JobState.CANCELLING, JobState.CANCELLED)


def test_terminal_states_are_immutable():
    assert is_terminal(JobState.SUCCEEDED)
    for state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
        with pytest.raises(ValueError, match="invalid job state transition"):
            validate_transition(state, JobState.QUEUED)


def test_cancelling_can_only_finish_as_cancelled():
    with pytest.raises(ValueError, match="invalid job state transition"):
        validate_transition(JobState.CANCELLING, JobState.FAILED)
