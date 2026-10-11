import random

import numpy as np
import pytest
import torch

from app.worker.runtime.reproducibility import configure_reproducibility


def test_configure_reproducibility_repeats_random_sequences():
    configure_reproducibility(1234)
    first = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    configure_reproducibility(1234)
    second = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize("deterministic", [True, False])
def test_configure_reproducibility_sets_deterministic_mode(deterministic):
    torch.use_deterministic_algorithms(not deterministic)

    configure_reproducibility(7, deterministic=deterministic)

    assert torch.are_deterministic_algorithms_enabled() is deterministic
