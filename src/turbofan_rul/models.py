"""Sequence regression models for RUL prediction."""

import math

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM encoder + linear head: reads a window of sensor readings, outputs one RUL value."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> outputs: (batch, seq_len, hidden_size)
        # h_n, c_n: final hidden and cell state of each layer, (num_layers, batch, hidden_size)
        outputs, (h_n, c_n) = self.lstm(x)
        last_hidden = outputs[:, -1, :]  # the state after reading the whole window
        return self.head(last_hidden).squeeze(-1)  # (batch,)


class GRURegressor(nn.Module):
    """GRU encoder + linear head; same interface as LSTMRegressor.

    A GRU has no separate cell state and 3 gate blocks instead of 4 (update, reset,
    candidate), so it has about 3/4 of the LSTM's parameters at the same hidden size.
    """

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> outputs: (batch, seq_len, hidden_size)
        outputs, h_n = self.gru(x)  # only a hidden state, no cell state
        return self.head(outputs[:, -1, :]).squeeze(-1)  # (batch,)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position signal added to every time step (Vaswani et al., 2017).

    Self-attention treats its inputs as an unordered set; this tells it which step is
    which. Even dimensions get sin, odd dimensions cos, at geometrically spaced
    frequencies. It is a buffer: saved with the model, never trained.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model); the same position vectors are added to every sample
        return x + self.encoding[: x.size(1)]


class TransformerRegressor(nn.Module):
    """Input projection -> positional encoding -> Transformer encoder -> mean pool -> head."""

    def __init__(self, n_features: int, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,  # pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> tokens: (batch, seq_len, d_model)
        tokens = self.positional_encoding(self.input_projection(x))
        # No causal mask: every step of the window is already in the past at prediction time.
        encoded = self.encoder(tokens)  # (batch, seq_len, d_model)
        pooled = encoded.mean(dim=1)    # average over the window: (batch, d_model)
        return self.head(pooled).squeeze(-1)  # (batch,)
