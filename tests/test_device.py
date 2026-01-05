import torch

from app.device import get_device


def test_cpu_device():
    device = get_device("cpu")
    assert device.type == "cpu"


def test_gpu_fallback_to_cpu():
    device = get_device("gpu")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"
