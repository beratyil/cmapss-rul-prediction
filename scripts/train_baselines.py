"""Train naive and linear-regression RUL baselines on FD001 and evaluate them.

Train/validation: engines from train_FD001 split by unit_id, every row has a target.
Test: the official protocol, one prediction at the last observed cycle of each engine.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from turbofan_rul.data import load_fd001
from turbofan_rul.metrics import mae, nasa_score, rmse
from turbofan_rul.preprocessing import FEATURE_COLUMNS, last_cycle_rows, split_engines
from turbofan_rul.targets import add_linear_rul

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def evaluate(model_name: str, split: str, y_true, y_pred) -> dict:
    row = {"model": model_name, "split": split, "n": len(y_true),
           "RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred)}
    # The NASA score is a sum, so it is only comparable on the fixed 100-engine test set.
    row["NASA score"] = nasa_score(y_true, y_pred) if split.startswith("test") else np.nan
    return row


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_predictions(val: pd.DataFrame, val_pred: np.ndarray, mean_rul: float,
                     y_test: np.ndarray, test_pred: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    for ax in axes:
        style_axes(ax)

    # 1) Every validation row: predicted vs true RUL.
    ax = axes[0]
    limit = val["rul"].max() + 10
    ax.scatter(val["rul"], val_pred, s=5, alpha=0.25, color=BLUE, linewidths=0)
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(limit * 0.97, limit * 0.9, "perfect", color=INK_MUTED, fontsize=8, ha="right")
    ax.set(xlim=(0, limit), ylim=(min(-20, val_pred.min() - 10), limit))
    ax.set_xlabel("True RUL (cycles)", color=INK_SECONDARY)
    ax.set_ylabel("Predicted RUL (cycles)", color=INK_SECONDARY)
    ax.set_title("Validation, all rows: linear regression", loc="left", color=INK)

    # 2) One validation engine over its whole life (the longest-lived one).
    ax = axes[1]
    longest = val.groupby("unit_id")["cycle"].max().idxmax()
    mask = (val["unit_id"] == longest).to_numpy()
    engine = val[mask]
    ax.plot(engine["cycle"], engine["rul"], color=INK_SECONDARY, linewidth=2,
            linestyle="--", label="True RUL")
    ax.plot(engine["cycle"], val_pred[mask], color=BLUE, linewidth=1.5, label="Linear regression")
    ax.axhline(mean_rul, color=ORANGE, linewidth=2, label="Naive (mean)")
    ax.set_xlabel("Cycle", color=INK_SECONDARY)
    ax.set_ylabel("RUL (cycles)", color=INK_SECONDARY)
    ax.set_title(f"Validation engine {longest} over its life", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 3) Official test protocol: one prediction per engine at its last observed cycle.
    ax = axes[2]
    limit = max(y_test.max(), test_pred.max()) + 15
    ax.scatter(y_test, test_pred, s=24, color=BLUE, edgecolors=SURFACE, linewidths=1)
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(limit * 0.97, limit * 0.9, "perfect", color=INK_MUTED, fontsize=8, ha="right")
    ax.text(limit * 0.03, limit * 0.93, "above line = late (riskier)", color=INK_MUTED, fontsize=8)
    ax.set(xlim=(0, limit), ylim=(min(-20, test_pred.min() - 10), limit))
    ax.set_xlabel("True RUL at last observed cycle", color=INK_SECONDARY)
    ax.set_ylabel("Predicted RUL", color=INK_SECONDARY)
    ax.set_title("Test, 100 engines: linear regression", loc="left", color=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    data = load_fd001(DATA_DIR)
    train, val = split_engines(add_linear_rul(data.train), val_fraction=0.2, seed=42)
    test = last_cycle_rows(data.test).merge(data.test_rul, on="unit_id")
    print(f"Engines -> train: {train['unit_id'].nunique()}, val: {val['unit_id'].nunique()}, "
          f"test: {test['unit_id'].nunique()}")
    print(f"Rows    -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # Scaling statistics come from the training engines only, then reused everywhere.
    scaler = StandardScaler().fit(train[FEATURE_COLUMNS])
    x_train = scaler.transform(train[FEATURE_COLUMNS])
    x_val = scaler.transform(val[FEATURE_COLUMNS])
    x_test = scaler.transform(test[FEATURE_COLUMNS])
    y_train = train["rul"].to_numpy()
    y_val = val["rul"].to_numpy()
    y_test = test["rul_at_last_observation"].to_numpy()

    mean_rul = y_train.mean()
    model = LinearRegression().fit(x_train, y_train)

    predictions = {
        "Naive (mean)": [np.full(len(y), mean_rul) for y in (y_train, y_val, y_test)],
        "Linear regression": [model.predict(x) for x in (x_train, x_val, x_test)],
    }
    rows = []
    for name, (train_pred, val_pred, test_pred) in predictions.items():
        rows.append(evaluate(name, "train (all rows)", y_train, train_pred))
        rows.append(evaluate(name, "val (all rows)", y_val, val_pred))
        rows.append(evaluate(name, "test (last cycle)", y_test, test_pred))
    print()
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    coefficients = pd.Series(model.coef_, index=FEATURE_COLUMNS).sort_values()
    print(f"\nIntercept: {model.intercept_:.2f}  (= mean training RUL {mean_rul:.2f})")
    print("Coefficients (RUL change per +1 std of the sensor):")
    print(coefficients.round(2).to_string())

    _, val_pred, test_pred = predictions["Linear regression"]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_predictions(val, val_pred, mean_rul, y_test, test_pred,
                     FIGURE_DIR / "fd001_baseline_predictions.png")
    print(f"\nFigure saved to {FIGURE_DIR / 'fd001_baseline_predictions.png'}")


if __name__ == "__main__":
    main()
