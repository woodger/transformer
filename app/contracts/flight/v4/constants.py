CONTRACT_NAME = "transformer-flight"
CONTRACT_VERSION = 4
CONTRACT_PATH_VERSION = "v4"

CAPABILITIES_ACTION = "transformer.v4.capabilities"
HEALTH_ACTION = "transformer.v4.health"
CREATE_ACTION = "transformer.v4.job.create"
ACQUIRE_ACTION = "transformer.v4.job.acquire"
STATUS_ACTION = "transformer.v4.job.status"
INPUTS_LIST_ACTION = "transformer.v4.job.inputs.list"
INPUT_CLOSE_ACTION = "transformer.v4.job.input.close"
OUTPUTS_LIST_ACTION = "transformer.v4.job.outputs.list"
CANCEL_ACTION = "transformer.v4.job.cancel"
MODEL_DESCRIBE_ACTION = "transformer.v4.model.describe"

ACTIONS = (
    CAPABILITIES_ACTION,
    HEALTH_ACTION,
    CREATE_ACTION,
    ACQUIRE_ACTION,
    STATUS_ACTION,
    INPUTS_LIST_ACTION,
    INPUT_CLOSE_ACTION,
    OUTPUTS_LIST_ACTION,
    CANCEL_ACTION,
    MODEL_DESCRIBE_ACTION,
)

FIT_SCHEMA_ID = "inventory.sequence.fit.v2"
PREDICT_SCHEMA_ID = "inventory.sequence.predict.v2"
PREDICTION_SCHEMA_ID = "transformer.prediction.target-aligned.v1"

# Inputs are listed with bounded revision-based pagination, so close carries a
# constant-size summary instead of one document entry per payload.
MAX_PAYLOADS_PER_JOB = 100_000
MAX_PAGE_ITEMS = 100
