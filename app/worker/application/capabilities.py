from __future__ import annotations

from typing import Protocol, cast

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.semantic.v5 import semantic_capabilities
from app.contracts.worker.v20 import (
    CHECKPOINT_FORMAT,
    CONTRACT_NAME,
    CONTRACT_VERSION,
    FIT_INPUT_SCHEMA_ID,
    PREDICT_INPUT_SCHEMA_ID,
    PREDICTION_OUTPUT_SCHEMA_ID,
    RECOVERY_FORMAT,
    validate_document,
)


class _CudaDeviceProperties(Protocol):
    name: str


def inspect_capabilities() -> JsonObject:
    """Проверить Torch/CUDA внутри границы процесса Worker."""

    import torch

    devices: list[JsonValue] = [{
        "backend": "cpu",
        "opaqueId": "cpu",
        "name": "CPU",
    }]
    if torch.cuda.is_available():
        for ordinal in range(torch.cuda.device_count()):
            properties = cast(
                _CudaDeviceProperties,
                torch.cuda.get_device_properties(  # pyright: ignore[reportUnknownMemberType]
                    ordinal
                ),
            )
            devices.append({
                "backend": "cuda",
                "opaqueId": f"cuda:{ordinal}",
                "name": properties.name,
            })
    document: JsonObject = {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "checkpointFormat": CHECKPOINT_FORMAT,
        "recoveryFormat": RECOVERY_FORMAT,
        "schemaIds": {
            "fitInput": FIT_INPUT_SCHEMA_ID,
            "predictInput": PREDICT_INPUT_SCHEMA_ID,
            "predictionOutput": PREDICTION_OUTPUT_SCHEMA_ID,
        },
        "semantic": semantic_capabilities(),
        "torchVersion": str(torch.__version__),
        "cudaRuntimeVersion": torch.version.cuda,
        "devices": devices,
    }
    return validate_document(document, "capabilities")


__all__ = ["inspect_capabilities"]
