CONTRACT_NAME = "transformer-flight"
CONTRACT_VERSION = 2
CONTRACT_PATH_VERSION = "v2"

CAPABILITIES_ACTION = "transformer.v2.capabilities"
HEALTH_ACTION = "transformer.v2.health"
CREATE_ACTION = "transformer.v2.job.create"
SEAL_ACTION = "transformer.v2.job.seal"
START_ACTION = "transformer.v2.job.start"
STATUS_ACTION = "transformer.v2.job.status"
CANCEL_ACTION = "transformer.v2.job.cancel"

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

# A complete seal manifest must fit in the 64 KiB action-document limit even
# with maximum-length idempotency keys and decimal ordinals. Four hundred
# entries leave an envelope margin and keep every advertised job sealable.
MAX_MANIFEST_ITEMS = 400

