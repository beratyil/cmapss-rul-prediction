"""Sequence regression models for RUL prediction."""

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
