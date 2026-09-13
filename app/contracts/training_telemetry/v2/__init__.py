"""Owner-scoped Training Telemetry Query revision 2 contract."""

from app.contracts.training_telemetry.v2.capabilities import (
    training_telemetry_capabilities,
)
from app.contracts.training_telemetry.v2.codec import (
    TrainingTelemetryContractError,
    validate_training_telemetry_document,
)
from app.contracts.training_telemetry.v2.constants import (
    CONTRACT_NAME,
    CONTRACT_REVISION,
    CURSOR_TTL_SECONDS,
    GRADIENT_INTERACTIONS_ACTION,
    MAX_EPOCH_PAGE_SIZE,
    MAX_GRADIENT_PAIR_PAGE_SIZE,
    MAX_RESPONSE_BYTES,
    MAX_RETAINED_SNAPSHOT_BYTES,
    MAX_RETAINED_SNAPSHOT_COUNT,
    MAX_RETAINED_SNAPSHOT_TOTAL_BYTES,
    REPORT_ACTION,
    SNAPSHOT_CAPACITY_RETRY_AFTER_SECONDS,
)

__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_REVISION",
    "CURSOR_TTL_SECONDS",
    "GRADIENT_INTERACTIONS_ACTION",
    "MAX_EPOCH_PAGE_SIZE",
    "MAX_GRADIENT_PAIR_PAGE_SIZE",
    "MAX_RESPONSE_BYTES",
    "MAX_RETAINED_SNAPSHOT_BYTES",
    "MAX_RETAINED_SNAPSHOT_COUNT",
    "MAX_RETAINED_SNAPSHOT_TOTAL_BYTES",
    "REPORT_ACTION",
    "SNAPSHOT_CAPACITY_RETRY_AFTER_SECONDS",
    "TrainingTelemetryContractError",
    "training_telemetry_capabilities",
    "validate_training_telemetry_document",
]
