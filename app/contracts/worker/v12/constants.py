CONTRACT_NAME = "transformer-worker"
CONTRACT_VERSION = 12
MAX_EVENT_BYTES = 1024 * 1024

CHECKPOINT_FORMAT = "transformer-checkpoint-v6"
RECOVERY_FORMAT = "transformer-recovery-v6"

FIT_INPUT_SCHEMA_ID = "transformer.indexed-feature-blocks.fit.v1"
PREDICT_INPUT_SCHEMA_ID = "transformer.indexed-feature-blocks.predict.v1"
PREDICTION_OUTPUT_SCHEMA_ID = "transformer.prediction.target-aligned.v3"

ARROW_SCHEMA_IDS = (
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
)

__all__ = [
    "ARROW_SCHEMA_IDS",
    "CHECKPOINT_FORMAT",
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "FIT_INPUT_SCHEMA_ID",
    "MAX_EVENT_BYTES",
    "PREDICTION_OUTPUT_SCHEMA_ID",
    "PREDICT_INPUT_SCHEMA_ID",
    "RECOVERY_FORMAT",
]

