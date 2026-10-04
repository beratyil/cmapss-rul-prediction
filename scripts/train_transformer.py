"""Small Transformer encoder for FD001 RUL (Milestone 7), compared with the earlier models.

Same windows (seq_len 50), piecewise target (cap 125), optimizer, early stopping and
seeds as compare_rnn.py, so only the architecture changes. Two sizes: d_model=32 has
about the RNNs' parameter budget, d_model=64 is roughly 3x larger. Nothing is tuned on
the test set here, so every run is tested once.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from turbofan_rul.data import load_fd001
from turbofan_rul.models import TransformerRegressor
from turbofan_rul.pipeline import evaluate_on_test, prepare_sequence_splits, validation_scorer
from turbofan_rul.preprocessing import FEATURE_COLUMNS
from turbofan_rul.targets import piecewise_rul
from turbofan_rul.training import fit, predict, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

SEQ_LEN = 50
SEEDS = [0, 1, 2, 3, 4]
RUL_CAP = 125
TEST_LIKE_MAX_RUL = 150
TRAINING = {"epochs": 80, "batch_size": 256, "lr": 1e-3, "patience": 10}

CANDIDATES = {
    "Transformer d=32": lambda n: TransformerRegressor(n, d_model=32, n_heads=4, n_layers=2,
                                                       dim_feedforward=64),
    "Transformer d=64": lambda n: TransformerRegressor(n, d_model=64, n_heads=4, n_layers=2,
                                                       dim_feedforward=128),
}
# Test RMSE of earlier experiments (compare_rnn.py, train_window_models.py).
REFERENCES = {"LSTM h=64, mean of 5 seeds": 15.12, "Gradient boosting, window 50": 14.08}

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = dict(zip(CANDIDATES, ["#2a78d6", "#eb6834"]))


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_results(runs: pd.DataFrame, curves: dict, best_models: dict, prepared: dict,
                 device: torch.device, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    for ax in axes:
        style_axes(ax)
    names = list(CANDIDATES)

    # 1) Test RMSE per seed and mean, against the earlier models.
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
    for (label, value), style in zip(REFERENCES.items(), (":", "--")):
        ax.axhline(value, color=INK_MUTED, linewidth=1.2, linestyle=style)
        ax.text(-0.45, value - 0.05, label, va="top", fontsize=8, color=INK_MUTED)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_xlim(-0.5, len(names) - 0.3)
    low = min(runs["test RMSE"].min(), *REFERENCES.values())
    ax.set_ylim(low - 0.6, runs["test RMSE"].max() + 0.4)
    ax.set_ylabel("Test RMSE (cycles), lower is better", color=INK_SECONDARY)
    ax.set_title("Test RMSE per seed (dots) and mean (line)", loc="left", color=INK)

    # 2) Validation curves, seed 0.
    ax = axes[1]
    for name in names:
        ax.plot(np.arange(1, len(curves[name]) + 1), curves[name], color=COLORS[name],
                linewidth=1.8, label=name)
    ax.set_xlabel("Epoch", color=INK_SECONDARY)
    ax.set_ylabel(f"Val RMSE, true RUL <= {TEST_LIKE_MAX_RUL}", color=INK_SECONDARY)
    ax.set_title("Validation curves, seed 0", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 3) Longest-lived validation engine, best-validation seed of each size.
    ax = axes[2]
    val_ends = prepared["val_ends"]
    longest = val_ends.groupby("unit_id")["cycle"].max().idxmax()
    mask = (val_ends["unit_id"] == longest).to_numpy()
    cycles, true_rul = val_ends["cycle"][mask], val_ends["rul"][mask]
    ax.plot(cycles, true_rul, color=INK_SECONDARY, linewidth=2, linestyle="--", label="True RUL")
    ax.plot(cycles, piecewise_rul(true_rul, RUL_CAP), color=INK_MUTED, linewidth=1.5,
            linestyle=":", label=f"Capped target ({RUL_CAP})")
    for name, model in best_models.items():
        pred = predict(model, prepared["x_val"][mask], device) * RUL_CAP
        ax.plot(cycles, pred, color=COLORS[name], linewidth=1.6, label=name)
    ax.set_xlabel("Cycle", color=INK_SECONDARY)
    ax.set_ylabel("RUL (cycles)", color=INK_SECONDARY)
    ax.set_title(f"Validation engine {longest} over its life", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    prepared = prepare_sequence_splits(load_fd001(DATA_DIR), SEQ_LEN, RUL_CAP)
    val_score = validation_scorer(prepared, device, RUL_CAP, TEST_LIKE_MAX_RUL)
    print(f"Input windows: train {prepared['x_train'].shape}, test {prepared['x_test'].shape}\n")

    # Warm-up so CUDA/cuDNN initialisation is not timed.
    for build in CANDIDATES.values():
        build(len(FEATURE_COLUMNS)).to(device)(
            torch.from_numpy(prepared["x_train"][:256]).to(device)).sum().backward()

    rows, curves, best_models = [], {}, {}
    for name, build in CANDIDATES.items():
        best_val = float("inf")
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
                         "best epoch": history.best_epoch,
                         "sec/epoch": history.seconds / epochs_run, "peak GPU MB": peak_mb,
                         "val RMSE": min(history.val_rmse),
                         **evaluate_on_test(model, prepared, device, RUL_CAP)})
            if seed == SEEDS[0]:
                curves[name] = history.val_rmse
            if min(history.val_rmse) < best_val:  # keep the best *validation* seed for plots
                best_val, best_models[name] = min(history.val_rmse), model

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
    for label, value in REFERENCES.items():
        print(f"Reference, {label}: test RMSE {value}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "fd001_transformer.png"
    plot_results(runs, curves, best_models, prepared, device, path)
    print(f"\nFigure saved to {path}")


if __name__ == "__main__":
    main()
