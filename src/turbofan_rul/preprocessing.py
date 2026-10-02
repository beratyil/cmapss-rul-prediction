"""Feature selection and engine-level data splitting for FD001."""

import numpy as np
import pandas as pd

# Screening result (scripts/screen_features_fd001.py): constant columns, sensor_6 and
# the noise-only operational settings 1-2 are removed.
FEATURE_COLUMNS = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]


def split_engines(
    frame: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by unit_id so that no engine appears in both train and validation."""
    engine_ids = frame["unit_id"].unique()
    rng = np.random.default_rng(seed)
    n_val = round(len(engine_ids) * val_fraction)
    val_ids = rng.choice(engine_ids, size=n_val, replace=False)

    is_val = frame["unit_id"].isin(val_ids)
    return frame[~is_val].copy(), frame[is_val].copy()


def last_cycle_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the final observed row of every engine: the test-set prediction point."""
    return frame.loc[frame.groupby("unit_id")["cycle"].idxmax()].reset_index(drop=True)
