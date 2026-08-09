from __future__ import annotations

from app.contracts.worker.v1 import (
    CONTRACT_NAME,
    CONTRACT_VERSION,
    validate_document,
)


def inspect_capabilities() -> dict:
    """Inspect Torch/CUDA inside the worker process boundary."""

    import torch

    devices = []
    if torch.cuda.is_available():
        for ordinal in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(ordinal)
            devices.append({
                "ordinal": ordinal,
                "name": properties.name,
            })
    document = {
        "contract": CONTRACT_NAME,
        "protocolVersion": CONTRACT_VERSION,
        "torchVersion": torch.__version__,
        "cudaRuntimeVersion": torch.version.cuda,
        "devices": devices,
    }
    return validate_document(document, "capabilities")


__all__ = ["inspect_capabilities"]
