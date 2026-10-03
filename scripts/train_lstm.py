"""Train a deliberately simple LSTM on FD001 windows and evaluate it like the baselines.

Inputs: windows of the 14 standardized sensors, shape (samples, seq_len, 14).
Target: piecewise RUL (cap 125) at the window's last cycle, divided by the cap.
Sequence length is chosen on validation engines (seed 0); the chosen setting is then
trained with three seeds to show run-to-run variance, and each is tested once.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from turbofan_rul.data import load_fd001
from turbofan_rul.metrics import mae, nasa_score, rmse
from turbofan_rul.models import LSTMRegressor
from turbofan_rul.preprocessing import FEATURE_COLUMNS, split_engines
from turbofan_rul.sequences import make_sequences
from turbofan_rul.targets import add_linear_rul, piecewise_rul
from turbofan_rul.training import fit, predict, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

SEQ_LENS = [30, 50]
SEEDS = [0, 1, 2]
RUL_CAP = 125
TEST_LIKE_MAX_RUL = 150
HIDDEN_SIZE = 64
TRAINING = {"epochs": 80, "batch_size": 256, "lr": 1e-3, "patience": 10}

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def prepare(data, seq_len: int) -> dict:
    """Engine split -> scaler fit on training engines -> windows for every split."""
    train, val = split_engines(add_linear_rul(data.train), val_fraction=0.2, seed=42)
    test = data.test.copy()
    scaler = StandardScaler().fit(train[FEATURE_COLUMNS])
    for frame in (train, val, test):
        frame[FEATURE_COLUMNS] = scaler.transform(frame[FEATURE_COLUMNS])

    x_train, train_ends = make_sequences(train, FEATURE_COLUMNS, seq_len)
    x_val, val_ends = make_sequences(val, FEATURE_COLUMNS, seq_len)
    x_test, test_ends = make_sequences(test, FEATURE_COLUMNS, seq_len, last_only=True)
    test_ends = test_ends.merge(data.test_rul, on="unit_id")
    y_train = piecewise_rul(train_ends["rul"].to_numpy(), RUL_CAP) / RUL_CAP
    return {"x_train": x_train, "y_train": y_train.astype(np.float32),
            "x_val": x_val, "val_ends": val_ends, "x_test": x_test, "test_ends": test_ends}


def train_one(prepared: dict, seed: int, device: torch.device):
    set_seed(seed)
    y_val = prepared["val_ends"]["rul"].to_numpy()
    test_like = y_val <= TEST_LIKE_MAX_RUL

    def val_score(model) -> float:
        pred = predict(model, prepared["x_val"], device) * RUL_CAP
        return rmse(y_val[test_like], pred[test_like])

    model = LSTMRegressor(n_features=len(FEATURE_COLUMNS), hidden_size=HIDDEN_SIZE).to(device)
    history = fit(model, prepared["x_train"], prepared["y_train"], val_score, device,
                  target_scale=RUL_CAP, **TRAINING)
    return model, history


def test_metrics(model, prepared: dict, device: torch.device) -> dict:
    y_test = prepared["test_ends"]["rul_at_last_observation"].to_numpy()
    pred = predict(model, prepared["x_test"], device) * RUL_CAP
    return {"test RMSE": rmse(y_test, pred), "test MAE": mae(y_test, pred),
            "NASA score": nasa_score(y_test, pred), "late predictions": int((pred > y_test).sum())}


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_results(history, model, prepared: dict, seq_len: int, seed: int,
                 device: torch.device, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    for ax in axes:
        style_axes(ax)

    # 1) Learning curves.
    ax = axes[0]
    epochs = np.arange(1, len(history.train_rmse) + 1)
    ax.plot(epochs, history.train_rmse, color=BLUE, linewidth=2, label="Train (vs capped target)")
    ax.plot(epochs, history.val_rmse, color=ORANGE, linewidth=2,
            label=f"Validation (true RUL <= {TEST_LIKE_MAX_RUL})")
    ax.axvline(history.best_epoch, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.text(history.best_epoch, max(history.val_rmse), " best epoch", color=INK_MUTED,
            fontsize=8, va="top")
    ax.set_xlabel("Epoch", color=INK_SECONDARY)
    ax.set_ylabel("RMSE (cycles)", color=INK_SECONDARY)
    ax.set_title(f"Training curve, seq_len {seq_len}, seed {seed}", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 2) Longest-lived validation engine.
    ax = axes[1]
    val_ends = prepared["val_ends"]
    longest = val_ends.groupby("unit_id")["cycle"].max().idxmax()
    mask = (val_ends["unit_id"] == longest).to_numpy()
    pred = predict(model, prepared["x_val"][mask], device) * RUL_CAP
    cycles = val_ends["cycle"][mask]
    ax.plot(cycles, val_ends["rul"][mask], color=INK_SECONDARY, linewidth=2, linestyle="--",
            label="True RUL")
    ax.plot(cycles, piecewise_rul(val_ends["rul"][mask], RUL_CAP), color=INK_MUTED,
            linewidth=1.5, linestyle=":", label=f"Capped target ({RUL_CAP})")
    ax.plot(cycles, pred, color=BLUE, linewidth=1.8, label="LSTM prediction")
    ax.set_xlabel("Cycle", color=INK_SECONDARY)
    ax.set_ylabel("RUL (cycles)", color=INK_SECONDARY)
    ax.set_title(f"Validation engine {longest} over its life", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # 3) Official test protocol.
    ax = axes[2]
    y_test = prepared["test_ends"]["rul_at_last_observation"].to_numpy()
    test_pred = predict(model, prepared["x_test"], device) * RUL_CAP
    limit = max(y_test.max(), test_pred.max()) + 15
    ax.scatter(y_test, test_pred, s=24, color=BLUE, edgecolors=SURFACE, linewidths=1)
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(limit * 0.97, limit * 0.9, "perfect", color=INK_MUTED, fontsize=8, ha="right")
    ax.text(limit * 0.03, limit * 0.93, "above line = late (riskier)", color=INK_MUTED, fontsize=8)
    ax.set(xlim=(0, limit), ylim=(-20, limit))
    ax.set_xlabel("True RUL at last observed cycle", color=INK_SECONDARY)
    ax.set_ylabel("Predicted RUL", color=INK_SECONDARY)
    ax.set_title("Test, 100 engines: LSTM", loc="left", color=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    data = load_fd001(DATA_DIR)

    # Step 1: choose the sequence length on validation engines (seed 0).
    prepared, runs, rows = {}, {}, []
    for seq_len in SEQ_LENS:
        prepared[seq_len] = prepare(data, seq_len)
        model, history = train_one(prepared[seq_len], SEEDS[0], device)
        runs[(seq_len, SEEDS[0])] = (model, history)
        n_params = sum(p.numel() for p in model.parameters())
        rows.append({"seq_len": seq_len, "train windows": len(prepared[seq_len]["x_train"]),
                     "input shape": tuple(prepared[seq_len]["x_train"].shape[1:]),
                     "parameters": n_params, "best epoch": history.best_epoch,
                     "val RMSE": min(history.val_rmse), "seconds": history.seconds})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    best_len = min(rows, key=lambda r: r["val RMSE"])["seq_len"]
    print(f"\nSelected seq_len: {best_len}")

    # Step 2: three seeds for the chosen length; each tested once.
    results = []
    for seed in SEEDS:
        if (best_len, seed) not in runs:
            runs[(best_len, seed)] = train_one(prepared[best_len], seed, device)
        model, history = runs[(best_len, seed)]
        results.append({"seed": seed, "best epoch": history.best_epoch,
                        "val RMSE": min(history.val_rmse),
                        **test_metrics(model, prepared[best_len], device)})
    results = pd.DataFrame(results)
    print()
    print(results.round(2).to_string(index=False))
    summary = results[["val RMSE", "test RMSE", "test MAE", "NASA score"]].agg(["mean", "std"])
    print("\nMean and std over seeds:")
    print(summary.round(2).to_string())
    print("Reference, gradient boosting (window 50): test RMSE 14.08, MAE 10.78, NASA 301")

    # Keep the seed with the best validation score (not the best test score).
    best_seed = int(results.loc[results["val RMSE"].idxmin(), "seed"])
    model, history = runs[(best_len, best_seed)]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "lstm_fd001.pt")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "fd001_lstm.png"
    plot_results(history, model, prepared[best_len], best_len, best_seed, device, path)
    print(f"\nModel saved to {MODEL_DIR / 'lstm_fd001.pt'}, figure saved to {path}")


if __name__ == "__main__":
    main()
