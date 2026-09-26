from app.contracts.json_types import JsonObject
from app.contracts.semantic.v5.constants import (
    DIRECT_OPERATORS,
    ENCODER_NORMALIZATION_ORDERS,
    MAX_OBJECTIVE_COMPONENTS,
    MAX_PRIVATE_RESOURCES,
    MAX_TARGET_SLOTS,
    OBJECTIVE_LANGUAGE_REVISION,
)


def semantic_capabilities() -> JsonObject:
    return {
        "objectiveLanguage": {
            "revision": OBJECTIVE_LANGUAGE_REVISION,
            "closed": True,
        },
        "semanticLimits": {
            "maxTargetSlots": MAX_TARGET_SLOTS,
            "maxObjectiveComponents": MAX_OBJECTIVE_COMPONENTS,
            "maxPrivateResources": MAX_PRIVATE_RESOURCES,
        },
        "directOperators": list(DIRECT_OPERATORS),
        "encoderNormalizationOrders": list(ENCODER_NORMALIZATION_ORDERS),
    }


__all__ = ["semantic_capabilities"]
