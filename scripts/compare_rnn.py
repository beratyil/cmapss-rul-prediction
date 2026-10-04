"""Fair LSTM vs GRU comparison on FD001 (Milestone 6).

Everything except the recurrent cell is shared: windows (seq_len 50), piecewise target
(cap 125), optimizer, batch size, early stopping and the five seeds. Two notions of a
fair size are tested: GRU with the LSTM's hidden size, and GRU with ~the LSTM's
parameter count. No setting is tuned here, so every run is evaluated on the test set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from turbofan_rul.data import load_fd001
from turbofan_rul.models import GRURegressor, LSTMRegressor
from turbofan_rul.pipeline import evaluate_on_test, prepare_sequence_splits, validation_scorer
from turbofan_rul.preprocessing import FEATURE_COLUMNS
from turbofan_rul.training import fit, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

SEQ_LEN = 50  # chosen on validation in train_lstm.py
SEEDS = [0, 1, 2, 3, 4]
RUL_CAP = 125
TEST_LIKE_MAX_RUL = 150
TRAINING = {"epochs": 80, "batch_size": 256, "lr": 1e-3, "patience": 10}
GRADIENT_BOOSTING_TEST_RMSE = 14.08  # train_window_models.py, window 50

CANDIDATES = {
    "LSTM h=64": lambda n: LSTMRegressor(n, hidden_size=64),
    "GRU h=64": lambda n: GRURegressor(n, hidden_size=64),
    "GRU h=75 (same params)": lambda n: GRURegressor(n, hidden_size=75),
}

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = dict(zip(CANDIDATES, ["#2a78d6", "#eb6834", "#1baf7a"]))


def warm_up(device: torch.device, x: np.ndarray) -> None:
    """One forward/backward per model type so CUDA/cuDNN set-up is not timed."""
    for build in CANDIDATES.values():
        model = build(len(FEATURE_COLUMNS)).to(device)
        model(torch.from_numpy(x[:256]).to(device)).sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_comparison(runs: pd.DataFrame, curves: dict, summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    for ax in axes:
        style_axes(ax)
    names = list(CANDIDATES)
    positions = np.arange(len(names))

    # 1) Test RMSE of every seed (dots) and the mean (line), vs gradient boosting.
    ax = axes[0]
    ax.grid(axis="x", visible=False)
    for i, name in enumerate(names):
        values = runs.loc[runs["model"] == name, "test RMSE"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(i + jitter, values, s=28, color=COLORS[name], edgecolors=SURFACE,
                   linewidths=1, zorder=3)
        ax.hlines(values.mean(), i - 0.22, i + 0.22, color=COLORS[name], linewidth=2.5)
        ax.text(i + 0.26, values.mean(), f"mean\n{values.mean():.2f}", va="center",
                fontsize=8, color=INK)
    ax.axhline(GRADIENT_BOOSTING_TEST_RMSE, color=INK_MUTED, linewidth=1.2, linestyle="--")
    ax.text(-0.45, GRADIENT_BOOSTING_TEST_RMSE - 0.05, "gradient boosting (window 50)",
            va="top", fontsize=8, color=INK_MUTED)
    ax.set_xticks(positions, names, fontsize=8)
    ax.set_xlim(-0.5, len(names) - 0.3)
    ax.set_ylim(GRADIENT_BOOSTING_TEST_RMSE - 0.6, runs["test RMSE"].max() + 0.4)
    ax.set_ylabel("Test RMSE (cycles), lower is better", color=INK_SECONDARY)
    ax.set_title("Test RMSE per seed (dots) and mean (line)", loc="left", color=INK)

    # 2) Validation curves of seed 0.
    ax = axes[1]
    for name in names:
        curve = curves[name]
        ax.plot(np.arange(1, len(curve) + 1), curve, color=COLORS[name], linewidth=1.8,
                label=name)
    ax.set_xlabel("Epoch", color=INK_SECONDARY)
    ax.set_ylabel(f"Val RMSE, true RUL <= {TEST_LIKE_MAX_RUL}", color=INK_SECONDARY)
    ax.set_title("Validation curves, seed 0", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 3) Cost: time per epoch, labelled with the parameter count.
    ax = axes[2]
    ax.grid(axis="x", visible=False)
    seconds = summary["sec/epoch"].to_numpy()
    ax.bar(positions, seconds, width=0.55, color=[COLORS[n] for n in names])
    for i, name in enumerate(names):
        ax.text(i, seconds[i] * 1.02, f"{int(summary.loc[name, 'parameters'])} params",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(positions, names, fontsize=8)
    ax.set_ylim(0, seconds.max() * 1.2)
    ax.set_ylabel("Training seconds per epoch (median)", color=INK_SECONDARY)
    ax.set_title("Training cost", loc="left", color=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    prepared = prepare_sequence_splits(load_fd001(DATA_DIR), SEQ_LEN, RUL_CAP)
    val_score = validation_scorer(prepared, device, RUL_CAP, TEST_LIKE_MAX_RUL)
    warm_up(device, prepared["x_train"])

    rows, curves = [], {}
    for name, build in CANDIDATES.items():
        for seed in SEEDS:
            set_seed(seed)
            model = build(len(FEATURE_COLUMNS)).to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            history = fit(model, prepared["x_train"], prepared["y_train"], val_score, device,
                          target_scale=RUL_CAP, **TRAINING)
            peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else np.nan
            epochs_run = len(history.val_rmse)
            rows.append({"model": name, "seed": seed,
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "best epoch": history.best_epoch, "epochs run": epochs_run,
                         "sec/epoch": history.seconds / epochs_run, "peak GPU MB": peak_mb,
                         "val RMSE": min(history.val_rmse),
                         **evaluate_on_test(model, prepared, device, RUL_CAP)})
            if seed == SEEDS[0]:
                curves[name] = history.val_rmse

    runs = pd.DataFrame(rows)
    print(runs.round(2).to_string(index=False))

    metrics = ["val RMSE", "test RMSE", "test MAE", "NASA score"]
    grouped = runs.groupby("model", sort=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    summary["parameters"] = grouped["parameters"].first()
    summary["sec/epoch"] = grouped["sec/epoch"].median()
    summary["peak GPU MB"] = grouped["peak GPU MB"].max()
    print("\nSummary over seeds (sec/epoch: median):")
    print(summary.round(2).to_string())
    print(f"Reference, gradient boosting (window 50): test RMSE {GRADIENT_BOOSTING_TEST_RMSE}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "fd001_lstm_vs_gru.png"
    plot_comparison(runs, curves, summary, path)
    print(f"\nFigure saved to {path}")


if __name__ == "__main__":
    main()
