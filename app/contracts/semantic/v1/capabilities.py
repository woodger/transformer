from app.contracts.json_types import JsonObject
from app.contracts.semantic.v1.constants import (
    AGGREGATIONS,
    AUXILIARY_OPERATORS,
    CONSTRAINTS,
    DIRECT_OPERATORS,
    MAX_OBJECTIVE_COMPONENTS,
    MAX_PRIVATE_RESOURCES,
    MAX_TARGET_SLOTS,
    MODEL_ARCHITECTURE_IDENTITY,
    MODEL_ARCHITECTURE_REVISION,
    OBJECTIVE_LANGUAGE_REVISION,
    REDUCTIONS,
    RESOURCE_KINDS,
    TRANSFORMATIONS,
)


def semantic_capabilities() -> JsonObject:
    return {
        "objectiveLanguage": {
            "revision": OBJECTIVE_LANGUAGE_REVISION,
            "constraints": list(CONSTRAINTS),
            "transformations": list(TRANSFORMATIONS),
            "resourceKinds": list(RESOURCE_KINDS),
            "directOperators": list(DIRECT_OPERATORS),
            "auxiliaryOperators": list(AUXILIARY_OPERATORS),
            "aggregations": list(AGGREGATIONS),
            "reductions": list(REDUCTIONS),
        },
        "semanticLimits": {
            "maxTargetSlots": MAX_TARGET_SLOTS,
            "maxObjectiveComponents": MAX_OBJECTIVE_COMPONENTS,
            "maxPrivateResources": MAX_PRIVATE_RESOURCES,
        },
        "modelArchitectures": [
            {
                "identity": MODEL_ARCHITECTURE_IDENTITY,
                "revision": MODEL_ARCHITECTURE_REVISION,
            }
        ],
    }


__all__ = ["semantic_capabilities"]

