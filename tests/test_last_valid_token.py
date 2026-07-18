import torch

from app.model.context import prepare_context_input
from app.model.transformer import TransformerModel, _last_unmasked_indices


def test_last_unmasked_indices_use_positions_not_valid_token_counts():
    key_padding_mask = torch.tensor([
        [True, False, False, True],
        [False, True, False, True],
        [True, False, True, True],
    ])

    indices = _last_unmasked_indices(key_padding_mask)

    assert indices.tolist() == [2, 2, 1]


def test_last_unmasked_index_uses_placeholder_for_all_missing_sequence():
    x = torch.full((1, 4, 2), float("nan"))

    _, key_padding_mask = prepare_context_input(x, "relaxed")

    assert key_padding_mask.tolist() == [[False, True, True, True]]
    assert _last_unmasked_indices(key_padding_mask).tolist() == [0]


def test_transformer_forward_with_leading_internal_and_all_missing_tokens_is_finite():
    model = TransformerModel(
        input_dim=2,
        seq_len=4,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        out_dim=6,
        nhead=4,
        context_mode="relaxed",
    )
    x = torch.randn(3, 4, 2)
    x[0, 0, :] = float("nan")
    x[1, 1, :] = float("nan")
    x[2, :, :] = float("nan")

    output = model(x)

    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()
