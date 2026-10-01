"""Check which FD001 columns carry information about engine degradation.

Life fraction = cycle / engine lifetime (near 0 = new, 1 = failure). It is used here
only as an analysis axis, not as a training target.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbofan_rul.data import OPERATIONAL_SETTING_COLUMNS, SENSOR_COLUMNS, load_cmapss_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

MEASUREMENT_COLUMNS = list(OPERATIONAL_SETTING_COLUMNS + SENSOR_COLUMNS)
LIFE_BINS = np.linspace(0, 1, 21)

# Reference palette: ink/chrome colors and the first categorical slots.
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"


def add_life_fraction(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cycle / lifetime per engine and a 20-bin version of it."""
    lifetime = frame.groupby("unit_id")["cycle"].transform("max")
    life_fraction = frame["cycle"] / lifetime
    life_bin = pd.cut(life_fraction, LIFE_BINS, include_lowest=True)
    return frame.assign(life_fraction=life_fraction, life_bin=life_bin)


def standardize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_life_correlation(
    correlations: pd.Series, constant_columns: list[str], highlight: list[str], path: Path
) -> None:
    ordered = correlations.reindex(correlations.abs().sort_values().index)
    colors = [BLUE if value >= 0 else RED for value in ordered]

    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(ordered.index, ordered.values, color=colors, height=0.7)
    ax.axvline(0, color=AXIS, linewidth=1)

    for name in highlight:
        value = ordered[name]
        offset = 0.02 if value >= 0 else -0.02
        ax.text(value + offset, name, f"{value:+.2f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9, color=INK)
    for label in ax.get_yticklabels():
        if label.get_text() in highlight:
            label.set_fontweight("bold")
            label.set_color(INK)

    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman correlation with life fraction (cycle / engine lifetime)",
                  color=INK_SECONDARY)
    ax.set_title("Which FD001 columns move as engines age?", loc="left",
                 fontsize=13, color=INK, pad=12)
    fig.text(0.01, 0.01, "Constant in training, correlation undefined: "
             + ", ".join(constant_columns), fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_low_variance_features(train: pd.DataFrame, rare_value: float, path: Path) -> None:
    bin_centers = (LIFE_BINS[:-1] + LIFE_BINS[1:]) / 2
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)

    style_axes(left)
    series = [("operational_setting_1", BLUE), ("operational_setting_2", ORANGE),
              ("sensor_11 (reference)", AQUA)]
    for label, color in series:
        column = label.split(" ")[0]
        binned = standardize(train[column]).groupby(train["life_bin"], observed=False).mean()
        left.plot(bin_centers, binned.values, color=color, linewidth=2,
                  marker="o", markersize=4, label=label)
    # Direct labels; the two settings overlap near 0, so they share one label.
    left.text(0.89, 1.9, "sensor_11", color=INK_SECONDARY, fontsize=8, ha="right")
    left.text(0.99, 0.12, "op settings 1 & 2", color=INK_SECONDARY, fontsize=8, ha="right")
    left.set_xlim(0, 1)
    left.set_xlabel("Life fraction", color=INK_SECONDARY)
    left.set_ylabel("Mean z-score in life bin", color=INK_SECONDARY)
    left.set_title("After standardization, noise stays flat", loc="left", color=INK)
    left.legend(frameon=False, fontsize=8, loc="upper left")

    style_axes(right)
    is_rare = train["sensor_6"].eq(rare_value)
    share = is_rare.groupby(train["life_bin"], observed=False).mean() * 100
    right.bar(bin_centers, share.values, width=0.045, color=BLUE)
    right.axhline(is_rare.mean() * 100, color=INK_MUTED, linewidth=1, linestyle="--")
    right.text(0.99, is_rare.mean() * 100 + 0.1, f"overall {is_rare.mean():.1%}",
               color=INK_MUTED, fontsize=8, ha="right", va="bottom")
    right.set_xlim(0, 1)
    right.set_xlabel("Life fraction", color=INK_SECONDARY)
    right.set_ylabel(f"Rows with sensor_6 = {rare_value:.2f} (%)", color=INK_SECONDARY)
    right.set_title(f"sensor_6: where the rare value {rare_value:.2f} appears",
                    loc="left", color=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    train = add_life_fraction(load_cmapss_split(DATA_DIR / "train_FD001.txt"))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    constant_columns = [c for c in MEASUREMENT_COLUMNS if train[c].nunique() == 1]
    print("Constant columns:", constant_columns)
    z_constant = standardize(train["sensor_1"])
    print(f"z-score of constant sensor_1 -> std={train['sensor_1'].std()}, "
          f"NaN rows={z_constant.isna().sum()} / {len(z_constant)}")

    print("\nScale invariance of correlation with life fraction (Pearson):")
    for column in ("operational_setting_1", "operational_setting_2"):
        raw = train[column].corr(train["life_fraction"])
        scaled = standardize(train[column]).corr(train["life_fraction"])
        print(f"  {column}: raw={raw:+.4f}  standardized={scaled:+.4f}")

    counts = train["sensor_6"].value_counts()
    rare_value = counts.idxmin()
    print("\nsensor_6 value counts:")
    print(counts.to_string())

    varying = [c for c in MEASUREMENT_COLUMNS if c not in constant_columns]
    correlations = train[varying].corrwith(train["life_fraction"], method="spearman")
    print("\nSpearman correlation with life fraction:")
    print(correlations.sort_values().round(3).to_string())

    highlight = ["operational_setting_1", "operational_setting_2", "sensor_6"]
    plot_life_correlation(correlations, constant_columns, highlight,
                          FIGURE_DIR / "fd001_feature_life_correlation.png")
    plot_low_variance_features(train, rare_value,
                               FIGURE_DIR / "fd001_low_variance_features.png")
    print(f"\nFigures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
