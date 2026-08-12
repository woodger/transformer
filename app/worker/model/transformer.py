import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import CONTEXT_MODE
from app.worker.model.context import (
    context_input_dim,
    prepare_context_input,
    validate_context_mode,
)
from app.worker.model.positional_encoding import PositionalEncoding


def _last_unmasked_indices(key_padding_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(
        key_padding_mask.size(1),
        device=key_padding_mask.device,
    ).expand_as(key_padding_mask)
    return positions.masked_fill(key_padding_mask, -1).max(dim=1).values.clamp(min=0)


class TradingHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        self.mean_head = nn.Linear(128, 1)
        self.sigma_head = nn.Linear(128, 1)
        self.return_scale_head = nn.Linear(128, 1)

        # Probability heads stay as logits inside the worker. The prediction
        # boundary converts them to target-space probabilities.
        self.ptp_head = nn.Linear(128, 1)
        self.psl_head = nn.Linear(128, 1)
        self.hit_head = nn.Linear(128, 1)

        self.vol_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)

        mean_return = torch.tanh(self.mean_head(h))
        sigma_return = torch.sigmoid(self.sigma_head(h))
        return_scale = F.softplus(self.return_scale_head(h)) + 1e-6

        take_profit_logit = self.ptp_head(h)
        stop_loss_logit = self.psl_head(h)
        hit_logit = self.hit_head(h)

        next_volatility = torch.sigmoid(self.vol_head(h))

        return torch.cat(
            [
                mean_return,
                sigma_return,
                take_profit_logit,
                stop_loss_logit,
                next_volatility,
                hit_logit,
                return_scale,
            ],
            dim=1
        )


class TransformerModel(nn.Module):
    """Map inputs to six public heads and one private uncertainty head.

    Context mode controls NaN masking, and the last unmasked timestep feeds
    the trading head for each sequence.
    """

    def __init__(
        self,
        input_dim,
        seq_len,
        hidden_dim,
        layers,
        dropout,
        out_dim,
        nhead=8,
        context_mode=CONTEXT_MODE,
    ):
        super().__init__()

        assert hidden_dim % nhead == 0

        self.input_dim = input_dim
        self.context_mode = validate_context_mode(context_mode)
        self.input_proj = nn.Linear(
            context_input_dim(input_dim, self.context_mode),
            hidden_dim,
        )
        self.pos = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.head = TradingHead(hidden_dim)

    def forward(self, x):
        x, key_padding_mask = prepare_context_input(x, self.context_mode)

        x = self.input_proj(x)
        x = self.pos(x)

        enc = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        last_unmasked_indices = _last_unmasked_indices(key_padding_mask)

        batch_idx = torch.arange(x.size(0), device=x.device)
        last_valid = enc[batch_idx, last_unmasked_indices]

        return self.head(last_valid)


def public_predictions(output: torch.Tensor) -> torch.Tensor:
    """Convert the worker's seven-head output to the six target-space values."""

    if output.ndim != 2 or output.shape[1] != 7:
        raise ValueError("model output must have shape [rows, 7]")
    return torch.stack(
        (
            output[:, 0],
            output[:, 1],
            torch.sigmoid(output[:, 2]),
            torch.sigmoid(output[:, 3]),
            output[:, 4],
            torch.sigmoid(output[:, 5]),
        ),
        dim=1,
    )
