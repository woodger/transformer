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
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")

        self.hidden_dim = hidden_dim
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map floating [batch, hidden] states to seven internal model heads."""

        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_dim:
            raise ValueError("hidden states must have shape [batch, hidden]")
        h = self.shared(hidden_states)

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
        input_dim: int,
        seq_len: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        out_dim: int,
        nhead: int = 8,
        context_mode: str = CONTEXT_MODE,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if layers <= 0:
            raise ValueError("layers must be a positive integer")
        if nhead <= 0 or hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by a positive nhead")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if out_dim <= 0:
            raise ValueError("out_dim must be a positive integer")

        self.input_dim = input_dim
        self.seq_len = seq_len
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Build seven internal heads from float32 model input.

        Args:
            features: Tensor [batch, sequence, features]. Sequence and feature
                dimensions must match the immutable model configuration. NaN
                values are interpreted according to ``context_mode``.

        Returns:
            Tensor [batch, 7]: six public target heads followed by the private
            Gaussian return scale. Probability heads remain logits here.
        """

        if features.ndim != 3:
            raise ValueError("features must have shape [batch, sequence, features]")
        if features.shape[1] != self.seq_len:
            raise ValueError("sequence length differs from model configuration")
        if features.shape[2] != self.input_dim:
            raise ValueError("feature dimension differs from model configuration")
        if features.dtype != torch.float32:
            raise ValueError("features must use float32")

        prepared_context = prepare_context_input(
            features,
            self.context_mode,
        )

        encoded_features = self.input_proj(prepared_context.features)
        encoded_features = self.pos(encoded_features)

        enc = self.encoder(
            encoded_features,
            src_key_padding_mask=prepared_context.key_padding_mask
        )

        last_unmasked_indices = _last_unmasked_indices(
            prepared_context.key_padding_mask
        )

        batch_idx = torch.arange(
            encoded_features.size(0),
            device=encoded_features.device,
        )
        last_valid = enc[batch_idx, last_unmasked_indices]

        return self.head(last_valid)


def public_predictions(model_output: torch.Tensor) -> torch.Tensor:
    """Convert the worker's seven-head output to the six target-space values."""

    if model_output.ndim != 2 or model_output.shape[1] != 7:
        raise ValueError("model output must have shape [rows, 7]")
    return torch.stack(
        (
            model_output[:, 0],
            model_output[:, 1],
            torch.sigmoid(model_output[:, 2]),
            torch.sigmoid(model_output[:, 3]),
            model_output[:, 4],
            torch.sigmoid(model_output[:, 5]),
        ),
        dim=1,
    )
