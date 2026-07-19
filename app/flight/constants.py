from enum import StrEnum


CONTRACT_NAME = "transformer-flight"
CONTRACT_VERSION = 1

CAPABILITIES_ACTION = "transformer.v1.capabilities"
HEALTH_ACTION = "transformer.v1.health"
CREATE_ACTION = "transformer.v1.job.create"
SEAL_ACTION = "transformer.v1.job.seal"
START_ACTION = "transformer.v1.job.start"
STATUS_ACTION = "transformer.v1.job.status"
CANCEL_ACTION = "transformer.v1.job.cancel"

ACTIONS = (
    CAPABILITIES_ACTION,
    HEALTH_ACTION,
    CREATE_ACTION,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
    CANCEL_ACTION,
)

FIT_SCHEMA_ID = "inventory.sequence.fit.v1"
PREDICT_SCHEMA_ID = "inventory.sequence.predict.v1"
PREDICTION_SCHEMA_ID = "transformer.prediction.v1"

SUPPORTED_OPERATIONS = ("fit", "predict")
SUPPORTED_DEVICES = ("cpu", "cuda", "auto")

# A complete seal manifest must fit in the 64 KiB action-document limit even
# with maximum-length idempotency keys and decimal ordinals.  Four hundred
# entries leave a conservative envelope margin and keep every advertised job
# sealable through the same public action contract.
MAX_MANIFEST_ITEMS = 400


class JobState(StrEnum):
    UPLOADING = "UPLOADING"
    SEALED = "SEALED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset({
    JobState.SUCCEEDED,
    JobState.FAILED,
    JobState.CANCELLED,
})

IMMEDIATE_CANCEL_STATES = frozenset({
    JobState.UPLOADING,
    JobState.SEALED,
    JobState.QUEUED,
})

STATE_TRANSITIONS = {
    JobState.UPLOADING: frozenset({JobState.SEALED, JobState.CANCELLED}),
    JobState.SEALED: frozenset({JobState.QUEUED, JobState.CANCELLED}),
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.CANCELLED}),
    JobState.RUNNING: frozenset({
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLING,
    }),
    JobState.CANCELLING: frozenset({JobState.CANCELLED}),
    JobState.SUCCEEDED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
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

    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    EXECUTION_INTERRUPTED = "EXECUTION_INTERRUPTED"
    SUBPROCESS_FAILED = "SUBPROCESS_FAILED"
    SUBPROCESS_HUNG = "SUBPROCESS_HUNG"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    CUDA_OUT_OF_MEMORY = "CUDA_OUT_OF_MEMORY"
    DISK_FULL = "DISK_FULL"
