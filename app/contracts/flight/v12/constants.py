from app.contracts.model_catalog.v1.constants import (
    DETAIL_ACTION as MODEL_CATALOG_DETAIL_ACTION,
    LIST_ACTION as MODEL_CATALOG_LIST_ACTION,
)

CONTRACT_NAME = "transformer-flight"
CONTRACT_VERSION = 12
CONTRACT_PATH_VERSION = "v12"

CAPABILITIES_ACTION = "transformer.v12.capabilities"
HEALTH_ACTION = "transformer.v12.health"
CREATE_ACTION = "transformer.v12.job.create"
ACQUIRE_ACTION = "transformer.v12.job.acquire"
STATUS_ACTION = "transformer.v12.job.status"
INPUTS_LIST_ACTION = "transformer.v12.job.inputs.list"
INPUT_CLOSE_ACTION = "transformer.v12.job.input.close"
OUTPUTS_LIST_ACTION = "transformer.v12.job.outputs.list"
CANCEL_ACTION = "transformer.v12.job.cancel"

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
    MODEL_CATALOG_LIST_ACTION,
    MODEL_CATALOG_DETAIL_ACTION,
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
    "MODEL_CATALOG_DETAIL_ACTION",
    "MODEL_CATALOG_LIST_ACTION",
    "OUTPUTS_LIST_ACTION",
    "PREDICTION_SCHEMA_ID",
    "PREDICT_SCHEMA_ID",
    "STATUS_ACTION",
]
