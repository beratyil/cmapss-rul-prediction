"""First look at the FD001 train/test files: shape, dtypes, missing values, statistics."""

from pathlib import Path

import pandas as pd

from turbofan_rul.data import load_cmapss_split, load_test_rul

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "CMAPSSData"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)


def inspect_split(name: str, frame: pd.DataFrame) -> None:
    """Print the basic structure and per-column statistics of one split."""
    print(f"\n{'=' * 25} {name} {'=' * 25}")
    print(frame.head())
    print()
    frame.info()

    print(f"\nMissing cells:   {int(frame.isna().sum().sum())}")
    print(f"Duplicated rows: {int(frame.duplicated().sum())}")

    summary = frame.describe().T
    summary.insert(0, "nunique", frame.nunique())
    print("\nPer-column summary:")
    print(summary.round(4))

    rows_per_engine = frame.groupby("unit_id").size()
    print(f"\nEngines: {rows_per_engine.size}")
    print("Cycles per engine:")
    print(rows_per_engine.describe().round(2))


def main() -> None:
    train = load_cmapss_split(DATA_DIR / "train_FD001.txt")
    test = load_cmapss_split(DATA_DIR / "test_FD001.txt")
    test_rul = load_test_rul(DATA_DIR / "RUL_FD001.txt")

    inspect_split("train_FD001", train)
    inspect_split("test_FD001", test)

    print(f"\n{'=' * 25} RUL_FD001 {'=' * 25}")
    print(test_rul["rul_at_last_observation"].describe().round(2))


if __name__ == "__main__":
    main()
