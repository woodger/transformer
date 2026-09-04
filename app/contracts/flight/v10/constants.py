CONTRACT_NAME = "transformer-flight"
CONTRACT_VERSION = 10
CONTRACT_PATH_VERSION = "v10"

CAPABILITIES_ACTION = "transformer.v10.capabilities"
HEALTH_ACTION = "transformer.v10.health"
CREATE_ACTION = "transformer.v10.job.create"
ACQUIRE_ACTION = "transformer.v10.job.acquire"
STATUS_ACTION = "transformer.v10.job.status"
INPUTS_LIST_ACTION = "transformer.v10.job.inputs.list"
INPUT_CLOSE_ACTION = "transformer.v10.job.input.close"
OUTPUTS_LIST_ACTION = "transformer.v10.job.outputs.list"
CANCEL_ACTION = "transformer.v10.job.cancel"
MODEL_DESCRIBE_ACTION = "transformer.v10.model.describe"

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

FIT_SCHEMA_ID = "transformer.indexed-feature-blocks.fit.v1"
PREDICT_SCHEMA_ID = "transformer.indexed-feature-blocks.predict.v1"
PREDICTION_SCHEMA_ID = "transformer.prediction.target-aligned.v3"

# Inputs are listed with bounded revision-based pagination, so close carries a
# constant-size summary instead of one document entry per payload.
MAX_PAYLOADS_PER_JOB = 100_000
MAX_PAGE_ITEMS = 100
