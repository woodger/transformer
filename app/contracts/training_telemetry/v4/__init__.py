"""Контракт owner-scoped Training Telemetry Query revision 4."""
from app.contracts.training_telemetry.v4.codec import (
    TrainingTelemetryContractError,
    validate_training_telemetry_document,
)
from app.contracts.training_telemetry.v4.constants import (
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
    "validate_training_telemetry_document",
]
