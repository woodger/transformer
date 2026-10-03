import pytest
import torch

from app.worker.model.positional_encoding import PositionalEncoding


def test_positional_encoding_preserves_sine_and_cosine_coordinates_for_odd_width():
    encoded = PositionalEncoding(3, max_len=2)(torch.zeros(1, 2, 3))

    torch.testing.assert_close(encoded, torch.tensor([[
        [0.0, 1.0, 0.0],
        [0.84147098, 0.54030231, 0.00215443],
    ]]), rtol=1e-5, atol=1e-8)


def test_positional_encoding_rejects_sequence_beyond_its_allocated_capacity():
    with pytest.raises(ValueError, match="positional encoding capacity"):
        PositionalEncoding(3, max_len=2)(torch.zeros(1, 3, 3))
