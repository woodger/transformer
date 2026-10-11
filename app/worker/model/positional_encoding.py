import numpy as np
import torch
import torch.nn as nn

DEFAULT_MAX_LEN = 5000


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = DEFAULT_MAX_LEN) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if max_len <= 0:
            raise ValueError("max_len must be a positive integer")

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])

        self.register_buffer("pe", pe)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Добавить позиции к признакам модели [batch, sequence, hidden]."""

        if features.ndim != 3:
            raise ValueError("model features must have shape [batch, sequence, hidden]")
        if features.shape[1] > self.pe.shape[0]:
            raise ValueError("sequence length exceeds positional encoding capacity")
        if features.shape[2] != self.pe.shape[1]:
            raise ValueError("model feature width differs from positional encoding")
        seq_len = features.size(1)
        return features + self.pe[:seq_len].unsqueeze(0)
