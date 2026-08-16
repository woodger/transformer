from __future__ import annotations

from typing import Protocol, cast

from app.contracts.json_types import JsonObject, JsonValue
from app.contracts.worker.v5 import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    validate_document,
)


class _CudaDeviceProperties(Protocol):
    name: str


def inspect_capabilities() -> JsonObject:
    """Inspect Torch/CUDA inside the worker process boundary."""

    import torch

    devices: list[JsonValue] = []
    if torch.cuda.is_available():
        for ordinal in range(torch.cuda.device_count()):
            properties = cast(
                _CudaDeviceProperties,
                torch.cuda.get_device_properties(  # pyright: ignore[reportUnknownMemberType]
                    ordinal
                ),
            )
            devices.append({
                "ordinal": ordinal,
                "name": properties.name,
            })
    document: JsonObject = {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "torchVersion": str(torch.__version__),
        "cudaRuntimeVersion": torch.version.cuda,
        "devices": devices,
    }
    return validate_document(document, "capabilities")


__all__ = ["inspect_capabilities"]
