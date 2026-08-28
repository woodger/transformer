from enum import StrEnum

SUPPORTED_OPERATIONS = ("fit", "predict")
SUPPORTED_DEVICES = ("cpu", "cuda", "auto")


class InputState(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    ABORTED = "ABORTED"


class ExecutionState(StrEnum):
    WAITING_INPUT = "WAITING_INPUT"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_EXECUTION_STATES = frozenset({
    ExecutionState.SUCCEEDED,
    ExecutionState.FAILED,
    ExecutionState.CANCELLED,
})

IMMEDIATE_CANCEL_STATES = frozenset({
    ExecutionState.WAITING_INPUT,
    ExecutionState.QUEUED,
    ExecutionState.RETRYING,
})

EXECUTION_STATE_TRANSITIONS: dict[
    ExecutionState,
    frozenset[ExecutionState],
] = {
    ExecutionState.WAITING_INPUT: frozenset({
        ExecutionState.QUEUED,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    }),
    ExecutionState.QUEUED: frozenset({
        ExecutionState.RUNNING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    }),
    ExecutionState.RUNNING: frozenset({
        ExecutionState.SUCCEEDED,
        ExecutionState.FAILED,
        ExecutionState.CANCELLING,
        ExecutionState.RETRYING,
    }),
    ExecutionState.RETRYING: frozenset({
        ExecutionState.RUNNING,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
    }),
    ExecutionState.CANCELLING: frozenset({ExecutionState.CANCELLED}),
    ExecutionState.SUCCEEDED: frozenset(),
    ExecutionState.FAILED: frozenset(),
    ExecutionState.CANCELLED: frozenset(),
}

INPUT_STATE_TRANSITIONS: dict[InputState, frozenset[InputState]] = {
    InputState.OPEN: frozenset({InputState.CLOSED, InputState.ABORTED}),
    InputState.CLOSED: frozenset(),
    InputState.ABORTED: frozenset(),
}


class ErrorCode(StrEnum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    CANCELLED = "CANCELLED"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"

    EMPTY_INPUT = "EMPTY_INPUT"
    STALE_FENCE = "STALE_FENCE"
    JOB_RETIRED = "JOB_RETIRED"
    INPUT_TIMEOUT = "INPUT_TIMEOUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_CORRUPT = "MODEL_CORRUPT"
    MODEL_SCHEMA_MISMATCH = "MODEL_SCHEMA_MISMATCH"

    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    DEVICE_LOST = "DEVICE_LOST"
    EXECUTION_INTERRUPTED = "EXECUTION_INTERRUPTED"
    SUBPROCESS_FAILED = "SUBPROCESS_FAILED"
    SUBPROCESS_HUNG = "SUBPROCESS_HUNG"
    WORKER_PROTOCOL_VIOLATION = "WORKER_PROTOCOL_VIOLATION"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    GPU_OUT_OF_MEMORY = "GPU_OUT_OF_MEMORY"
    DISK_FULL = "DISK_FULL"
    RECOVERY_CHECKPOINT_UNAVAILABLE = "RECOVERY_CHECKPOINT_UNAVAILABLE"
    RECOVERY_CHECKPOINT_INCOMPATIBLE = "RECOVERY_CHECKPOINT_INCOMPATIBLE"
    RECOVERY_INPUT_UNAVAILABLE = "RECOVERY_INPUT_UNAVAILABLE"
