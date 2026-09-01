import torch

from app import config as defaults


def get_device(device_arg: str | None = None) -> torch.device:
    dev = device_arg or defaults.DEFAULT_DEVICE

    if dev == "cpu":
        return torch.device("cpu")

    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dev == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")

    raise ValueError(f"Unsupported device: {dev}")
