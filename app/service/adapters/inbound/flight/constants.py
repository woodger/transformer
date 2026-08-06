"""Flight v2 wire constants combined with service lifecycle values."""

from app.contracts.flight.v2.constants import (
    ACTIONS,
    CANCEL_ACTION,
    CAPABILITIES_ACTION,
    CONTRACT_NAME,
    CONTRACT_PATH_VERSION,
    CONTRACT_VERSION,
    CREATE_ACTION,
    FIT_SCHEMA_ID,
    HEALTH_ACTION,
    MAX_MANIFEST_ITEMS,
    PREDICT_SCHEMA_ID,
    PREDICTION_SCHEMA_ID,
    SEAL_ACTION,
    START_ACTION,
    STATUS_ACTION,
)
from app.service.domain.job import (
    SUPPORTED_DEVICES,
    SUPPORTED_OPERATIONS,
    TERMINAL_STATES,
    ErrorCode,
    JobState,
)

__all__ = [
    "ACTIONS",
    "CANCEL_ACTION",
    "CAPABILITIES_ACTION",
    "CONTRACT_NAME",
    "CONTRACT_PATH_VERSION",
    "CONTRACT_VERSION",
    "CREATE_ACTION",
    "FIT_SCHEMA_ID",
    "HEALTH_ACTION",
    "MAX_MANIFEST_ITEMS",
    "PREDICTION_SCHEMA_ID",
    "PREDICT_SCHEMA_ID",
    "SEAL_ACTION",
    "START_ACTION",
    "STATUS_ACTION",
    "SUPPORTED_DEVICES",
    "SUPPORTED_OPERATIONS",
    "TERMINAL_STATES",
    "ErrorCode",
    "JobState",
]

