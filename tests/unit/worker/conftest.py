import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(1729)
            np.random.seed(1729)
            torch.manual_seed(1729)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark = cudnn_benchmark
