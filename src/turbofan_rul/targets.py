"""Remaining Useful Life (RUL) targets for run-to-failure trajectories."""

import pandas as pd


def add_linear_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `rul` = cycles left until the engine's last observed cycle.

    Only valid for run-to-failure data (the training file), where the last cycle of
    every engine is its failure cycle. Test trajectories stop before failure.
    """
    last_cycle = frame.groupby("unit_id")["cycle"].transform("max")
    return frame.assign(rul=last_cycle - frame["cycle"])
