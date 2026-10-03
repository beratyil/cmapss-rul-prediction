"""Turn per-engine time series into fixed-length windows for sequence models."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _engine_windows(values: np.ndarray, seq_len: int) -> np.ndarray:
    """Windows ending at every cycle of one engine: shape (n_cycles, seq_len, n_features).

    Cycles with fewer than `seq_len - 1` predecessors are front-padded by repeating the
    engine's first observation ("before we started observing, it looked like this").
    Zero padding would mean "fleet average" after standardization, a fake jump.
    """
    padding = np.repeat(values[:1], seq_len - 1, axis=0)
    padded = np.concatenate([padding, values], axis=0)
    windows = sliding_window_view(padded, window_shape=seq_len, axis=0)  # (n, features, seq_len)
    return windows.transpose(0, 2, 1)


def make_sequences(
    frame: pd.DataFrame, feature_columns: list[str], seq_len: int, last_only: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack windows from every engine, never mixing engines inside a window.

    Returns X with shape (n_samples, seq_len, n_features) as float32, and the frame rows
    the windows end at (unit_id, cycle, targets...), in the same order. With
    `last_only=True` only each engine's final window is kept (the test protocol).
    """
    frame = frame.sort_values(["unit_id", "cycle"])
    windows, ends = [], []
    for _, engine in frame.groupby("unit_id", sort=True):
        engine_windows = _engine_windows(engine[feature_columns].to_numpy(np.float32), seq_len)
        if last_only:
            engine_windows, engine = engine_windows[-1:], engine.iloc[-1:]
        windows.append(engine_windows)
        ends.append(engine)
    return np.concatenate(windows), pd.concat(ends).reset_index(drop=True)
