"""Reusable loaders for the NASA C-MAPSS text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = ("unit_id", "cycle")
OPERATIONAL_SETTING_COLUMNS = tuple(f"operational_setting_{index}" for index in range(1, 4))
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
CMAPSS_COLUMNS = IDENTIFIER_COLUMNS + OPERATIONAL_SETTING_COLUMNS + SENSOR_COLUMNS


@dataclass(frozen=True)
class FD001Data:
    """The three files that define the FD001 training and test data."""

    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.DataFrame


def _read_numeric_table(path: Path, column_names: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")

    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(column_names):
        raise ValueError(
            f"Expected {len(column_names)} columns in {path.name}, found {frame.shape[1]}."
        )

    frame.columns = column_names
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric value found in {path.name}.") from error

    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite value found in {path.name}.")
    return frame


def load_cmapss_split(path: str | Path) -> pd.DataFrame:
    """Load one C-MAPSS train/test split with named, consistently typed columns."""

    frame = _read_numeric_table(Path(path), CMAPSS_COLUMNS)

    for column in IDENTIFIER_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integers.")
        if (values <= 0).any():
            raise ValueError(f"{column} must contain positive values.")
        frame[column] = frame[column].astype("int64")

    measurement_columns = OPERATIONAL_SETTING_COLUMNS + SENSOR_COLUMNS
    frame[list(measurement_columns)] = frame[list(measurement_columns)].astype("float64")
    return frame


def load_test_rul(path: str | Path) -> pd.DataFrame:
    """Load the one endpoint RUL value supplied for each test engine."""

    path = Path(path)
    frame = _read_numeric_table(path, ("rul_at_last_observation",))
    values = frame["rul_at_last_observation"].to_numpy(dtype=float)
    if not np.equal(values, np.floor(values)).all() or (values < 0).any():
        raise ValueError(f"RUL values must be nonnegative integers in {path.name}.")

    return pd.DataFrame(
        {
            "unit_id": np.arange(1, len(frame) + 1, dtype=np.int64),
            "rul_at_last_observation": values.astype("int64"),
        }
    )


def load_fd001(data_directory: str | Path) -> FD001Data:
    """Load FD001 train, test, and test endpoint RUL files from one directory."""

    data_directory = Path(data_directory)
    return FD001Data(
        train=load_cmapss_split(data_directory / "train_FD001.txt"),
        test=load_cmapss_split(data_directory / "test_FD001.txt"),
        test_rul=load_test_rul(data_directory / "RUL_FD001.txt"),
    )
