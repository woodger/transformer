"""Owner-scoped Training Telemetry Query revision 1 contract."""

from app.contracts.training_telemetry.v1.capabilities import (
    training_telemetry_capabilities,
)
from app.contracts.training_telemetry.v1.codec import (
    TrainingTelemetryContractError,
    validate_training_telemetry_document,
)
from app.contracts.training_telemetry.v1.constants import (
    CONTRACT_NAME,
    CONTRACT_REVISION,
    CURSOR_TTL_SECONDS,
    GRADIENT_INTERACTIONS_ACTION,
    MAX_EPOCH_PAGE_SIZE,
    MAX_GRADIENT_PAIR_PAGE_SIZE,
    MAX_RESPONSE_BYTES,
    REPORT_ACTION,
)

__all__ = [
    "CONTRACT_NAME",
    "CONTRACT_REVISION",
    "CURSOR_TTL_SECONDS",
    "GRADIENT_INTERACTIONS_ACTION",
    "MAX_EPOCH_PAGE_SIZE",
    "MAX_GRADIENT_PAIR_PAGE_SIZE",
    "MAX_RESPONSE_BYTES",
    "REPORT_ACTION",
    "TrainingTelemetryContractError",
    "training_telemetry_capabilities",
    "validate_training_telemetry_document",
]
