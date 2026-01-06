import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding


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

        assert hidden_dim % nhead == 0, "hidden_dim must be divisible by nhead"

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
            encoder_layer, num_layers=layers
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, out_dim)
        )

    # def forward(self, x):
    #     x = self.input_proj(x)
    #     x = self.pos(x)
    #     enc = self.encoder(x)
    #     return self.head(enc[:, -1])

    def forward(self, x):
        # x: (B, S, input_dim)

        # mask: True = игнорировать позицию
        key_padding_mask = torch.isnan(x).any(dim=-1)  # (B, S)

        # убрать NaN из данных
        x = torch.nan_to_num(x, nan=0.0)

        # обычный pipeline
        x = self.input_proj(x)
        x = self.pos(x)

        # attention mask
        enc = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        # prediction по последнему валидному токену
        return self.head(enc[:, -1])
