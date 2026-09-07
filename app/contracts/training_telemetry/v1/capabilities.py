from app.contracts.json_types import JsonObject
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


def training_telemetry_capabilities() -> JsonObject:
    return {
        "contract": CONTRACT_NAME,
        "revision": CONTRACT_REVISION,
        "operations": ["report", "gradientInteractions"],
        "transport": {
            "kind": "ArrowFlightDoAction",
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
            "kind": "PreOptimizerUpdateEpochPass",
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
        },
        "features": {
            "ownerScope": "AuthenticatedSubject",
            "selector": "ExactModelRef",
            "batchQuery": False,
            "partialReports": False,
            "checkpointVerification": False,
            "structuredErrors": True,
        },
    }


__all__ = ["training_telemetry_capabilities"]
