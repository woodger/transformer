import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import CONTEXT_MODE
from app.model.context import context_input_dim, prepare_context_input, validate_context_mode
from app.model.positional_encoding import PositionalEncoding


class TradingHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        # regression
        self.mean_head = nn.Linear(128, 1)
        self.sigma_head = nn.Linear(128, 1)

        # logits (ВАЖНО: без sigmoid!)
        self.ptp_head = nn.Linear(128, 1)
        self.psl_head = nn.Linear(128, 1)
        self.hit_head = nn.Linear(128, 1)

        # positive regression
        self.vol_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)

        meanR = torch.tanh(self.mean_head(h))             # [-1, 1]
        sigmaR = F.softplus(self.sigma_head(h)) + 1e-6    # > 0

        logitTP = self.ptp_head(h)                         # logits
        logitSL = self.psl_head(h)                         # logits
        logitHit = self.hit_head(h)                        # logits

        volNext = F.softplus(self.vol_head(h)) + 1e-6      # > 0

        return torch.cat(
            [meanR, sigmaR, logitTP, logitSL, volNext, logitHit],
            dim=1
        )


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        hidden_dim,
        layers,
        dropout,
        out_dim,   # можно оставить, но он теперь логически = 6
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

        # последний валидный токен
        lengths = (~key_padding_mask).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        batch_idx = torch.arange(x.size(0), device=x.device)
        last_valid = enc[batch_idx, lengths]

        return self.head(last_valid)
