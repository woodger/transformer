CONTRACT_NAME = "transformer-flight"
CONTRACT_VERSION = 11
CONTRACT_PATH_VERSION = "v11"

CAPABILITIES_ACTION = "transformer.v11.capabilities"
HEALTH_ACTION = "transformer.v11.health"
CREATE_ACTION = "transformer.v11.job.create"
ACQUIRE_ACTION = "transformer.v11.job.acquire"
STATUS_ACTION = "transformer.v11.job.status"
INPUTS_LIST_ACTION = "transformer.v11.job.inputs.list"
INPUT_CLOSE_ACTION = "transformer.v11.job.input.close"
OUTPUTS_LIST_ACTION = "transformer.v11.job.outputs.list"
CANCEL_ACTION = "transformer.v11.job.cancel"
MODEL_DESCRIBE_ACTION = "transformer.v11.model.describe"

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

MAX_PAYLOADS_PER_JOB = 100_000
MAX_PAGE_ITEMS = 100

__all__ = [
    "ACQUIRE_ACTION",
    "ACTIONS",
    "CANCEL_ACTION",
    "CAPABILITIES_ACTION",
    "CONTRACT_NAME",
    "CONTRACT_PATH_VERSION",
    "CONTRACT_VERSION",
    "CREATE_ACTION",
    "FIT_SCHEMA_ID",
    "HEALTH_ACTION",
    "INPUTS_LIST_ACTION",
    "INPUT_CLOSE_ACTION",
    "MAX_PAGE_ITEMS",
    "MAX_PAYLOADS_PER_JOB",
    "MODEL_DESCRIBE_ACTION",
    "OUTPUTS_LIST_ACTION",
    "PREDICTION_SCHEMA_ID",
    "PREDICT_SCHEMA_ID",
    "STATUS_ACTION",
]

