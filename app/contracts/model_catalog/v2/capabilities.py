from app.contracts.json_types import JsonObject
from app.contracts.model_catalog.v2.constants import (
    CONTRACT_NAME,
    CONTRACT_REVISION,
    CURSOR_TTL_SECONDS,
    DETAIL_ACTION,
    LIST_ACTION,
    MAX_CHECKPOINT_VERIFICATION_BYTES,
    MAX_PAGE_SIZE,
    MAX_RESPONSE_BYTES,
)


def catalog_capabilities() -> JsonObject:
    return {
        "contract": CONTRACT_NAME,
        "revision": CONTRACT_REVISION,
        "operations": ["detail", "list"],
        "transport": {
            "protocol": "ArrowFlightDoAction",
            "actions": {
                "detail": DETAIL_ACTION,
                "list": LIST_ACTION,
            },
        },
        "consistency": {
            "consistencyModel": "LiveHighWater",
            "ordering": [
                {"field": "createdAt", "direction": "DESC"},
                {"field": "modelRef", "direction": "ASC"},
            ],
            "concurrentPublication": "ExcludedAfterHighWater",
            "concurrentDeletion": "MayOmit",
            "cursorExpiration": "RestartTraversal",
        },
        "limits": {
            "maxPageSize": MAX_PAGE_SIZE,
            "cursorTtlSeconds": CURSOR_TTL_SECONDS,
            "maxResponseBytes": MAX_RESPONSE_BYTES,
            "maxCheckpointVerificationsPerDetail": 1,
            "maxCheckpointVerificationBytes": (
                MAX_CHECKPOINT_VERIFICATION_BYTES
            ),
        },
        "features": {
            "ownerScope": "AuthenticatedSubject",
            "filters": False,
            "batchDetail": False,
            "serverSideComparison": False,
            "checkpointVerification": "FullSha256",
            "structuredErrors": True,
        },
    }


__all__ = ["catalog_capabilities"]
