import torch
from config import DEFAULT_DEVICE


def get_device(device_arg: str | None = None) -> torch.device:
    dev = device_arg or DEFAULT_DEVICE

    if dev == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
