"""Training loop shared by the sequence models (LSTM now, GRU and Transformer later)."""

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device,
            batch_size: int = 2048) -> np.ndarray:
    model.eval()  # no dropout / training-only behaviour
    batches = [model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu()
               for i in range(0, len(x), batch_size)]
    return torch.cat(batches).numpy()


@dataclass
class History:
    train_rmse: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0


def fit(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_score: Callable[[nn.Module], float],
    device: torch.device,
    target_scale: float = 1.0,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
) -> History:
    """Mini-batch Adam on MSE with early stopping; restores the best validation weights.

    `val_score(model)` returns the validation error in cycles (lower is better), so the
    caller decides which validation rows count. `target_scale` converts the training
    loss back to cycles for logging.
    """
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = History()
    best_score, best_state, epochs_without_gain = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        squared_error, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()            # clear gradients of the previous batch
            loss = loss_fn(model(xb), yb)    # forward pass
            loss.backward()                  # backpropagation (through time)
            optimizer.step()                 # one weight update per mini-batch
            squared_error += loss.item() * len(xb)
            n_seen += len(xb)

        score = val_score(model)
        history.train_rmse.append(float(np.sqrt(squared_error / n_seen)) * target_scale)
        history.val_rmse.append(score)
        if score < best_score:
            best_score, best_state, epochs_without_gain = score, copy.deepcopy(model.state_dict()), 0
            history.best_epoch = epoch
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                break

    model.load_state_dict(best_state)
    history.seconds = time.perf_counter() - start
    return history
