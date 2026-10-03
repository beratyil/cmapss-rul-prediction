"""Classical models with history features: Ridge and gradient boosting on FD001.

For each window length, every sensor gets its raw value plus trailing mean, std and
slope. Window length (and Ridge's alpha) are chosen on validation engines; the test set
is used once, for the best window of each model family.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from turbofan_rul.data import load_fd001
from turbofan_rul.features import add_window_features
from turbofan_rul.metrics import mae, nasa_score, rmse
from turbofan_rul.preprocessing import FEATURE_COLUMNS, last_cycle_rows, split_engines
from turbofan_rul.targets import add_linear_rul, piecewise_rul

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

WINDOWS = [1, 10, 20, 30, 40, 50]  # 1 = current row only, no history
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000]
RUL_CAP = 125  # selected in compare_rul_targets.py
TEST_LIKE_MAX_RUL = 150  # documented test RUL range, see compare_rul_targets.py

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FAMILY_COLORS = {"Ridge": BLUE, "Gradient boosting": ORANGE}


def build_splits(data, window: int):
    """Window features per engine, then engine split; test keeps each engine's last row."""
    if window == 1:
        train_all, columns = add_linear_rul(data.train), list(FEATURE_COLUMNS)
        test_all = data.test
    else:
        train_all, columns = add_window_features(add_linear_rul(data.train), FEATURE_COLUMNS, window)
        test_all, _ = add_window_features(data.test, FEATURE_COLUMNS, window)
    train, val = split_engines(train_all, val_fraction=0.2, seed=42)
    # Features are computed on the full test history first, then the last row is kept.
    test = last_cycle_rows(test_all).merge(data.test_rul, on="unit_id")
    return train, val, test, columns


def fit_models(x_train, y_train, x_val, y_val, test_like):
    """Return {family: (fitted model, validation RMSE, label)} for one window length."""
    def val_rmse(model):
        return rmse(y_val[test_like], model.predict(x_val)[test_like])

    ridges = [Ridge(alpha=a).fit(x_train, y_train) for a in RIDGE_ALPHAS]
    scores = [val_rmse(m) for m in ridges]
    best = int(np.argmin(scores))

    # early_stopping=False: its internal split is by row and would mix engines.
    boosting = HistGradientBoostingRegressor(early_stopping=False, random_state=42)
    boosting.fit(x_train, y_train)

    return {
        "Ridge": (ridges[best], scores[best], f"alpha={RIDGE_ALPHAS[best]}"),
        "Gradient boosting": (boosting, val_rmse(boosting), "default"),
    }


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_results(sweep: pd.DataFrame, best: dict, engine_view: dict, test_view: dict,
                 path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    for ax in axes:
        style_axes(ax)

    # 1) Validation RMSE vs window length per model family.
    ax = axes[0]
    for family, color in FAMILY_COLORS.items():
        part = sweep[sweep["model"] == family]
        ax.plot(part["window"], part["val RMSE"], color=color, linewidth=2, marker="o",
                markersize=5, label=family)
        window = best[family]
        score = part.loc[part["window"] == window, "val RMSE"].item()
        ax.scatter(window, score, s=140, facecolors="none", edgecolors=INK, linewidths=1.5,
                   zorder=3)
    ax.set_xticks(WINDOWS, ["1\n(no history)"] + [str(w) for w in WINDOWS[1:]])
    ax.set_xlabel("Window length (cycles)", color=INK_SECONDARY)
    ax.set_ylabel(f"Val RMSE, true RUL <= {TEST_LIKE_MAX_RUL} (cycles)", color=INK_SECONDARY)
    ax.set_title("Choosing the window on validation engines", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 2) Longest-lived validation engine: no history vs best windows.
    ax = axes[1]
    cycles, true_rul = engine_view["cycle"], engine_view["true"]
    ax.plot(cycles, true_rul, color=INK_SECONDARY, linewidth=2, linestyle="--", label="True RUL")
    ax.plot(cycles, engine_view["no history"], color=INK_MUTED, linewidth=1.2,
            label="Ridge, window 1")
    for family, color in FAMILY_COLORS.items():
        ax.plot(cycles, engine_view[family], color=color, linewidth=1.8,
                label=f"{family}, window {best[family]}")
    ax.set_xlabel("Cycle", color=INK_SECONDARY)
    ax.set_ylabel("RUL (cycles)", color=INK_SECONDARY)
    ax.set_title(f"Validation engine {engine_view['unit_id']} over its life", loc="left",
                 color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 3) Official test protocol.
    ax = axes[2]
    y_test = test_view["true"]
    limit = max(y_test.max(), *(test_view[f].max() for f in FAMILY_COLORS)) + 15
    for family, color in FAMILY_COLORS.items():
        ax.scatter(y_test, test_view[family], s=24, color=color, edgecolors=SURFACE,
                   linewidths=1, label=f"{family}, window {best[family]}")
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(limit * 0.97, limit * 0.9, "perfect", color=INK_MUTED, fontsize=8, ha="right")
    ax.set(xlim=(0, limit), ylim=(-20, limit))
    ax.set_xlabel("True RUL at last observed cycle", color=INK_SECONDARY)
    ax.set_ylabel("Predicted RUL", color=INK_SECONDARY)
    ax.set_title("Test, 100 engines", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    data = load_fd001(DATA_DIR)
    sweep_rows, runs = [], {}

    # Step 1: for each window, fit both families and score them on validation only.
    for window in WINDOWS:
        train, val, test, columns = build_splits(data, window)
        scaler = StandardScaler().fit(train[columns])
        x_train, x_val, x_test = (scaler.transform(f[columns]) for f in (train, val, test))
        y_train = piecewise_rul(train["rul"].to_numpy(), RUL_CAP)
        y_val = val["rul"].to_numpy()
        test_like = y_val <= TEST_LIKE_MAX_RUL

        for family, (model, score, note) in fit_models(x_train, y_train, x_val, y_val,
                                                       test_like).items():
            sweep_rows.append({"window": window, "model": family, "features": len(columns),
                               "val RMSE": score, "setting": note})
            runs[(family, window)] = (model, val, x_val, test, x_test)

    sweep = pd.DataFrame(sweep_rows)
    print(sweep.round(2).to_string(index=False))

    # Step 2: best window per family on validation, then a single test evaluation.
    best = {f: int(sweep.loc[sweep[sweep["model"] == f]["val RMSE"].idxmin(), "window"])
            for f in FAMILY_COLORS}
    results, test_view, engine_view = [], {}, {}
    candidates = [("Ridge", 1)] + [(f, best[f]) for f in FAMILY_COLORS]
    for family, window in candidates:
        model, val, x_val, test, x_test = runs[(family, window)]
        y_test = test["rul_at_last_observation"].to_numpy()
        test_pred = model.predict(x_test)
        errors = test_pred - y_test
        results.append({"model": family, "window": window, "test RMSE": rmse(y_test, test_pred),
                        "test MAE": mae(y_test, test_pred),
                        "NASA score": nasa_score(y_test, test_pred),
                        "late predictions": int((errors > 0).sum())})

        longest = val.groupby("unit_id")["cycle"].max().idxmax()
        mask = (val["unit_id"] == longest).to_numpy()
        key = "no history" if window == 1 else family
        engine_view.update({"unit_id": longest, "cycle": val["cycle"][mask].to_numpy(),
                            "true": val["rul"][mask].to_numpy(), key: model.predict(x_val)[mask]})
        if window > 1:
            test_view.update({"true": y_test, family: test_pred})

    print(f"\nBest window on validation: {best}\n")
    print(pd.DataFrame(results).round(2).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "fd001_window_models.png"
    plot_results(sweep, best, engine_view, test_view, path)
    print(f"\nFigure saved to {path}")


if __name__ == "__main__":
    main()
