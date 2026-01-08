import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encoding import PositionalEncoding


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
        self.ptp_head = nn.Linear(128, 1)
        self.psl_head = nn.Linear(128, 1)
        self.vol_head = nn.Linear(128, 1)
        self.hit_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)

        meanR = torch.tanh(self.mean_head(h))            # [-1, 1]
        sigmaR = F.softplus(self.sigma_head(h)) + 1e-6   # > 0

        pTP = torch.sigmoid(self.ptp_head(h))            # [0, 1]
        pSL = torch.sigmoid(self.psl_head(h))            # [0, 1]

        volNext = F.softplus(self.vol_head(h)) + 1e-6    # > 0
        hitTP = torch.sigmoid(self.hit_head(h))          # [0, 1]

        return torch.cat(
            [meanR, sigmaR, pTP, pSL, volNext, hitTP],
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
        out_dim,
        nhead=8,
    ):
        super().__init__()

        assert hidden_dim % nhead == 0

        self.input_proj = nn.Linear(input_dim, hidden_dim)
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

        # заменяем flat-head на специализированный
        self.head = TradingHead(hidden_dim)

    def forward(self, x):
        # x: (B, S, input_dim)

        key_padding_mask = torch.isnan(x).any(dim=-1)  # True = ignore
        x = torch.nan_to_num(x, nan=0.0)

        x = self.input_proj(x)
        x = self.pos(x)

        enc = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        # берем последний валидный токен
        lengths = (~key_padding_mask).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        batch_idx = torch.arange(x.size(0), device=x.device)
        last_valid = enc[batch_idx, lengths]

        return self.head(last_valid)
