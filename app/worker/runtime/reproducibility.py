import random

import numpy as np
import torch


def configure_reproducibility(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(bool(deterministic))
    if deterministic:
        torch.backends.cudnn.benchmark = False
