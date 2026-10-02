"""Compare linear vs piecewise (capped) RUL targets with linear regression on FD001.

The cap is a hyperparameter chosen on validation engines. Every evaluation uses the
true (uncapped) RUL, because that is the quantity we ultimately want to predict.
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
from turbofan_rul.targets import add_linear_rul, piecewise_rul

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

CAPS = [None, 150, 130, 125, 115, 100, 90]
# Saxena et al. (2008) document test RULs between 10 and 150 cycles. Validation rows in
# that range mimic the test protocol without looking at any test label.
TEST_LIKE_MAX_RUL = 150

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def fit_predict(cap: float | None, x_train, y_train, *x_eval) -> list[np.ndarray]:
    """Fit linear regression on the (optionally capped) target and predict each input."""
    target = y_train if cap is None else piecewise_rul(y_train, cap)
    model = LinearRegression().fit(x_train, target)
    return [model.predict(x) for x in x_eval]


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_comparison(sweep: pd.DataFrame, best_index: int, val: pd.DataFrame,
                    val_preds: dict, y_test: np.ndarray, test_preds: dict,
                    best_cap: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    for ax in axes:
        style_axes(ax)
    linear_name, capped_name = list(val_preds)

    # 1) Validation RMSE for each cap, under both validation protocols.
    ax = axes[0]
    positions = np.arange(len(sweep))
    all_rows, test_like = sweep.columns[1], sweep.columns[2]
    ax.plot(positions, sweep[all_rows], color=INK_MUTED, linewidth=2, marker="o",
            markersize=5, label=all_rows)
    ax.plot(positions, sweep[test_like], color=AQUA, linewidth=2, marker="o",
            markersize=5, label=test_like)
    ax.scatter(positions[best_index], sweep[test_like].iloc[best_index], s=140,
               facecolors="none", edgecolors=INK, linewidths=1.5, zorder=3)
    ax.annotate("selected", (positions[best_index], sweep[test_like].iloc[best_index]),
                xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=8, color=INK)
    ax.set_xticks(positions, sweep["cap"].astype(str))
    ax.set_xlabel("RUL cap (none = linear target)", color=INK_SECONDARY)
    ax.set_ylabel("Validation RMSE vs true RUL (cycles)", color=INK_SECONDARY)
    ax.set_title("Choosing the cap on validation engines", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # 2) The longest-lived validation engine: what each target teaches the model.
    ax = axes[1]
    longest = val.groupby("unit_id")["cycle"].max().idxmax()
    mask = (val["unit_id"] == longest).to_numpy()
    engine = val[mask]
    ax.plot(engine["cycle"], engine["rul"], color=INK_SECONDARY, linewidth=2,
            linestyle="--", label="True RUL")
    ax.plot(engine["cycle"], piecewise_rul(engine["rul"], best_cap), color=INK_MUTED,
            linewidth=1.5, linestyle=":", label=f"Capped target ({best_cap})")
    ax.plot(engine["cycle"], val_preds[linear_name][mask], color=BLUE, linewidth=1.5,
            label=f"Pred. ({linear_name})")
    ax.plot(engine["cycle"], val_preds[capped_name][mask], color=ORANGE, linewidth=1.5,
            label=f"Pred. ({capped_name})")
    ax.set_xlabel("Cycle", color=INK_SECONDARY)
    ax.set_ylabel("RUL (cycles)", color=INK_SECONDARY)
    ax.set_title(f"Validation engine {longest} over its life", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 3) Official test protocol, both targets, against the true RUL.
    ax = axes[2]
    limit = max(y_test.max(), *(p.max() for p in test_preds.values())) + 15
    for (name, pred), color in zip(test_preds.items(), (BLUE, ORANGE)):
        ax.scatter(y_test, pred, s=24, color=color, edgecolors=SURFACE, linewidths=1,
                   label=name)
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(limit * 0.97, limit * 0.9, "perfect", color=INK_MUTED, fontsize=8, ha="right")
    ax.set(xlim=(0, limit), ylim=(-20, limit))
    ax.set_xlabel("True RUL at last observed cycle", color=INK_SECONDARY)
    ax.set_ylabel("Predicted RUL", color=INK_SECONDARY)
    ax.set_title("Test, 100 engines: linear regression", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    data = load_fd001(DATA_DIR)
    train, val = split_engines(add_linear_rul(data.train), val_fraction=0.2, seed=42)
    test = last_cycle_rows(data.test).merge(data.test_rul, on="unit_id")

    scaler = StandardScaler().fit(train[FEATURE_COLUMNS])
    x_train, x_val, x_test = (scaler.transform(f[FEATURE_COLUMNS]) for f in (train, val, test))
    y_train, y_val = train["rul"].to_numpy(), val["rul"].to_numpy()
    y_test = test["rul_at_last_observation"].to_numpy()
    test_like = y_val <= TEST_LIKE_MAX_RUL

    # Step 1: choose the cap on validation engines only.
    all_rows_col = "val RMSE, all rows"
    test_like_col = f"val RMSE, true RUL <= {TEST_LIKE_MAX_RUL}"
    sweep_rows = []
    for cap in CAPS:
        (val_pred,) = fit_predict(cap, x_train, y_train, x_val)
        sweep_rows.append({
            "cap": "none" if cap is None else cap,
            all_rows_col: rmse(y_val, val_pred),
            test_like_col: rmse(y_val[test_like], val_pred[test_like]),
        })
    sweep = pd.DataFrame(sweep_rows)
    print(f"Validation rows: {len(y_val)} total, {test_like.sum()} with true RUL <= "
          f"{TEST_LIKE_MAX_RUL}\n")
    print(sweep.round(2).to_string(index=False))

    best_index = int(sweep[test_like_col].idxmin())
    best_cap = CAPS[best_index]
    print(f"\nSelected cap (lowest test-like validation RMSE): {best_cap}")

    # Step 2: evaluate the linear and the selected capped target once on the test set.
    candidates = {"linear target": None, f"cap {best_cap}": best_cap}
    val_preds, test_preds, rows = {}, {}, []
    for name, cap in candidates.items():
        val_pred, test_pred = fit_predict(cap, x_train, y_train, x_val, x_test)
        val_preds[name], test_preds[name] = val_pred, test_pred
        errors = test_pred - y_test
        rows.append({"target": name, "test RMSE": rmse(y_test, test_pred),
                     "test MAE": mae(y_test, test_pred),
                     "NASA score": nasa_score(y_test, test_pred),
                     "late predictions": int((errors > 0).sum()),
                     "mean error (pred - true)": errors.mean()})
    print()
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "fd001_capped_rul_comparison.png"
    plot_comparison(sweep, best_index, val, val_preds, y_test, test_preds, best_cap, path)
    print(f"\nFigure saved to {path}")


if __name__ == "__main__":
    main()
