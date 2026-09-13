from app.contracts.json_types import JsonObject
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
)


def training_telemetry_capabilities() -> JsonObject:
    return {
        "contract": CONTRACT_NAME,
        "revision": CONTRACT_REVISION,
        "operations": ["report", "gradientInteractions"],
        "transport": {
            "protocol": "ArrowFlightDoAction",
            "actions": {
                "report": REPORT_ACTION,
                "gradientInteractions": GRADIENT_INTERACTIONS_ACTION,
            },
        },
        "metricFamilies": [
            "epochLosses",
            "targetErrors",
            "optimizerHealth",
            "gradientInteractions",
        ],
        "availabilityStates": ["pending", "unavailable", "available"],
        "gradientAvailabilityStates": [
            "notConfigured",
            "configuredWithoutObservations",
            "available",
        ],
        "epochObservation": {
            "measurementPoint": "PreOptimizerUpdateEpochPass",
            "checkpointReevaluation": False,
        },
        "outcomePrecedence": [
            "ModelLookup",
            "Pending",
            "Unavailable",
            "IntegrityFailure",
            "Available",
        ],
        "limits": {
            "maxEpochPageSize": MAX_EPOCH_PAGE_SIZE,
            "maxGradientPairPageSize": MAX_GRADIENT_PAIR_PAGE_SIZE,
            "cursorTtlSeconds": CURSOR_TTL_SECONDS,
            "maxResponseBytes": MAX_RESPONSE_BYTES,
            "maxRetainedSnapshotCount": MAX_RETAINED_SNAPSHOT_COUNT,
            "maxRetainedSnapshotBytes": MAX_RETAINED_SNAPSHOT_BYTES,
            "maxRetainedSnapshotTotalBytes": (MAX_RETAINED_SNAPSHOT_TOTAL_BYTES),
        },
        "features": {
            "ownerScope": "AuthenticatedSubject",
            "selector": "ExactModelRef",
            "batchQuery": False,
            "partialReports": False,
            "checkpointVerification": False,
            "structuredErrors": True,
            "cursorRestartOutcome": "TELEMETRY_CURSOR_INVALIDATED",
            "snapshotAdmission": "AtomicNoLiveEviction",
        },
    }


__all__ = ["training_telemetry_capabilities"]
