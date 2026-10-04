"""Data preparation and evaluation shared by the FD001 sequence models."""

from typing import Callable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from turbofan_rul.data import FD001Data
from turbofan_rul.metrics import mae, nasa_score, rmse
from turbofan_rul.preprocessing import FEATURE_COLUMNS, split_engines
from turbofan_rul.sequences import make_sequences
from turbofan_rul.targets import add_linear_rul, piecewise_rul
from turbofan_rul.training import predict


def prepare_sequence_splits(data: FD001Data, seq_len: int, rul_cap: float) -> dict:
    """Engine split -> scaler fit on training engines -> windows for every split.

    y_train is the piecewise RUL divided by `rul_cap`, so targets lie in [0, 1].
    """
    train, val = split_engines(add_linear_rul(data.train), val_fraction=0.2, seed=42)
    test = data.test.copy()
    scaler = StandardScaler().fit(train[FEATURE_COLUMNS])
    for frame in (train, val, test):
        frame[FEATURE_COLUMNS] = scaler.transform(frame[FEATURE_COLUMNS])

    x_train, train_ends = make_sequences(train, FEATURE_COLUMNS, seq_len)
    x_val, val_ends = make_sequences(val, FEATURE_COLUMNS, seq_len)
    x_test, test_ends = make_sequences(test, FEATURE_COLUMNS, seq_len, last_only=True)
    test_ends = test_ends.merge(data.test_rul, on="unit_id")
    y_train = piecewise_rul(train_ends["rul"].to_numpy(), rul_cap) / rul_cap
    return {"x_train": x_train, "y_train": y_train.astype(np.float32),
            "x_val": x_val, "val_ends": val_ends, "x_test": x_test, "test_ends": test_ends}


def validation_scorer(prepared: dict, device: torch.device, rul_cap: float,
                      max_rul: float) -> Callable[[nn.Module], float]:
    """RMSE in cycles on validation windows whose true RUL <= max_rul (the test-like rows)."""
    y_val = prepared["val_ends"]["rul"].to_numpy()
    test_like = y_val <= max_rul

    def score(model: nn.Module) -> float:
        pred = predict(model, prepared["x_val"], device) * rul_cap
        return rmse(y_val[test_like], pred[test_like])

    return score


def evaluate_on_test(model: nn.Module, prepared: dict, device: torch.device,
                     rul_cap: float) -> dict:
    """Official protocol: one prediction per test engine at its last observed cycle."""
    y_test = prepared["test_ends"]["rul_at_last_observation"].to_numpy()
    pred = predict(model, prepared["x_test"], device) * rul_cap
    return {"test RMSE": rmse(y_test, pred), "test MAE": mae(y_test, pred),
            "NASA score": nasa_score(y_test, pred), "late predictions": int((pred > y_test).sum())}
