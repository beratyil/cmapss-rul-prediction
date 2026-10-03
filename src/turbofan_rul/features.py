"""History features: trailing-window statistics computed per engine."""

import pandas as pd


def _rolling(values: pd.DataFrame, unit_id: pd.Series, window: int, stat: str) -> pd.DataFrame:
    """Trailing `stat` over the last `window` rows of the same engine (shorter at the start)."""
    rolled = getattr(values.groupby(unit_id).rolling(window, min_periods=1), stat)()
    return rolled.reset_index(level=0, drop=True).loc[values.index]


def add_window_features(
    frame: pd.DataFrame, columns: list[str], window: int
) -> tuple[pd.DataFrame, list[str]]:
    """Add trailing mean, std and least-squares slope of each column over `window` cycles.

    A window only contains the current and earlier cycles of the same engine. Early
    cycles use the rows available so far instead of padding, so engines with a short
    history (like test engines observed for 31 cycles) still get valid features.
    Returns the new frame and the full feature list (raw values + window statistics).
    """
    frame = frame.sort_values(["unit_id", "cycle"])
    values = frame[columns]
    unit_id = frame["unit_id"]
    t = pd.DataFrame({c: frame["cycle"].astype(float) for c in columns}, index=frame.index)

    mean = _rolling(values, unit_id, window, "mean")
    std = _rolling(values, unit_id, window, "std").fillna(0.0)  # one row -> no spread

    # Slope of x over cycle t: cov(t, x) / var(t), built from rolling means.
    mean_t = _rolling(t, unit_id, window, "mean")
    var_t = _rolling(t * t, unit_id, window, "mean") - mean_t**2
    cov_tx = _rolling(t * values, unit_id, window, "mean") - mean_t * mean
    slope = (cov_tx / var_t).where(var_t > 0, 0.0)  # one row -> no trend yet

    stats = {"mean": mean, "std": std, "slope": slope}
    new_columns = {f"{c}_{name}{window}": table[c] for name, table in stats.items() for c in columns}
    return frame.assign(**new_columns), list(columns) + list(new_columns)
