"""Focused validation, summaries, and static figures for FD001 exploration."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from turbofan_rul.data import SENSOR_COLUMNS


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FIGURE_DPI = 160
COLORS = ("#0072B2", "#D55E00", "#009E73")


def split_summary(frame: pd.DataFrame) -> pd.Series:
    """Summarize rows, engines, and observed trajectory lengths."""

    maximum_cycles = frame.groupby("unit_id")["cycle"].max()
    return pd.Series(
        {
            "rows": len(frame),
            "columns": frame.shape[1],
            "engines": frame["unit_id"].nunique(),
            "minimum_observed_cycles": maximum_cycles.min(),
            "median_observed_cycles": maximum_cycles.median(),
            "mean_observed_cycles": maximum_cycles.mean(),
            "maximum_observed_cycles": maximum_cycles.max(),
        }
    )


def validate_split(frame: pd.DataFrame) -> pd.Series:
    """Report structural problems without modifying the data."""

    numeric_values = frame.to_numpy(dtype=float)
    starts_at_one = 0
    nonconsecutive = 0
    for _, engine in frame.groupby("unit_id", sort=False):
        cycles = engine["cycle"].to_numpy()
        if cycles[0] != 1:
            starts_at_one += 1
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            nonconsecutive += 1

    return pd.Series(
        {
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicated_rows": int(frame.duplicated().sum()),
            "duplicated_unit_cycle_pairs": int(
                frame.duplicated(subset=["unit_id", "cycle"]).sum()
            ),
            "non_finite_cells": int((~np.isfinite(numeric_values)).sum()),
            "non_positive_unit_ids": int((frame["unit_id"] <= 0).sum()),
            "non_positive_cycles": int((frame["cycle"] <= 0).sum()),
            "engines_not_starting_at_cycle_1": starts_at_one,
            "engines_with_nonconsecutive_cycles": nonconsecutive,
        }
    )


def column_profile(
    frame: pd.DataFrame, near_constant_dominance: float = 0.95
) -> pd.DataFrame:
    """Profile uniqueness and variance using an explicit near-constant rule."""

    records: list[dict[str, object]] = []
    for column in frame.columns:
        counts = frame[column].value_counts(dropna=False, normalize=True)
        unique_values = int(frame[column].nunique(dropna=False))
        dominant_ratio = float(counts.iloc[0])
        is_constant = unique_values == 1
        records.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype),
                "unique_values": unique_values,
                "minimum": float(frame[column].min()),
                "maximum": float(frame[column].max()),
                "variance": float(frame[column].var()),
                "dominant_value": counts.index[0],
                "dominant_ratio": dominant_ratio,
                "is_constant": is_constant,
                "is_near_constant": (
                    not is_constant and dominant_ratio >= near_constant_dominance
                ),
            }
        )
    return pd.DataFrame.from_records(records).set_index("column")


def representative_engine_ids(frame: pd.DataFrame) -> tuple[int, int, int]:
    """Choose engines closest to minimum, median, and maximum lifetime."""

    lifetimes = frame.groupby("unit_id")["cycle"].max()
    targets = (lifetimes.min(), lifetimes.median(), lifetimes.max())
    return tuple(int((lifetimes - target).abs().idxmin()) for target in targets)


def sensor_trend_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe within-engine monotonic association with cycle for each sensor."""

    records: list[dict[str, float | int | str]] = []
    for sensor in SENSOR_COLUMNS:
        correlations: list[float] = []
        for _, engine in frame.groupby("unit_id", sort=False):
            if engine[sensor].nunique() > 1:
                correlation = engine[sensor].rank().corr(engine["cycle"].rank())
                if pd.notna(correlation):
                    correlations.append(float(correlation))

        if correlations:
            values = np.asarray(correlations)
            records.append(
                {
                    "sensor": sensor,
                    "engines_with_variation": len(values),
                    "median_spearman_rho": float(np.median(values)),
                    "q1_spearman_rho": float(np.percentile(values, 25)),
                    "q3_spearman_rho": float(np.percentile(values, 75)),
                }
            )
        else:
            records.append(
                {
                    "sensor": sensor,
                    "engines_with_variation": 0,
                    "median_spearman_rho": np.nan,
                    "q1_spearman_rho": np.nan,
                    "q3_spearman_rho": np.nan,
                }
            )

    summary = pd.DataFrame.from_records(records).set_index("sensor")
    return summary.sort_values(
        "median_spearman_rho", key=lambda values: values.abs(), ascending=False
    )


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_engine_lifetimes(frame: pd.DataFrame, path: Path) -> Path:
    lifetimes = frame.groupby("unit_id")["cycle"].max()
    bins = np.arange(120, 381, 20)
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.hist(lifetimes, bins=bins, color=COLORS[0], edgecolor="white", alpha=0.9)
    axis.axvline(
        lifetimes.mean(),
        color=COLORS[1],
        linestyle="--",
        label=f"Mean: {lifetimes.mean():.1f}",
    )
    axis.axvline(
        lifetimes.median(),
        color=COLORS[2],
        linestyle=":",
        linewidth=2,
        label=f"Median: {lifetimes.median():.0f}",
    )
    axis.set(
        title="FD001 Training Engine Lifetime Distribution",
        xlabel="Lifetime (operational cycles)",
        ylabel="Number of engines",
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    return _save_figure(figure, path)


def plot_operational_settings(frame: pd.DataFrame, path: Path) -> Path:
    engine_ids = representative_engine_ids(frame)
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    for index, axis in enumerate(axes, start=1):
        column = f"operational_setting_{index}"
        for color, unit_id in zip(COLORS, engine_ids, strict=True):
            engine = frame.loc[frame["unit_id"] == unit_id]
            axis.plot(
                engine["cycle"],
                engine[column],
                color=color,
                linewidth=1.1,
                label=f"Engine {unit_id}",
            )
        axis.set(ylabel=f"Setting {index}")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Operational cycle")
    figure.suptitle("FD001 Operational Settings for Representative Engines", y=1.01)
    figure.tight_layout()
    return _save_figure(figure, path)


def plot_sensor_distributions(
    frame: pd.DataFrame, sensors: list[str], path: Path
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for axis, sensor in zip(axes.flat, sensors, strict=True):
        axis.hist(frame[sensor], bins=35, color=COLORS[0], edgecolor="white", alpha=0.9)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.set(
            title=sensor.replace("_", " ").title(),
            xlabel="Sensor value",
            ylabel="Observations",
        )
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle("FD001 Training Sensor Distributions (Selected Sensors)", y=1.01)
    figure.tight_layout()
    return _save_figure(figure, path)


def plot_sensor_variance(frame: pd.DataFrame, path: Path) -> Path:
    constant = [sensor for sensor in SENSOR_COLUMNS if frame[sensor].nunique() == 1]
    nonconstant = [sensor for sensor in SENSOR_COLUMNS if sensor not in constant]
    variances = frame[nonconstant].var().sort_values()
    figure, axis = plt.subplots(figsize=(9, 6.2))
    axis.barh(variances.index, variances.values, color=COLORS[0], alpha=0.9)
    axis.set_xscale("log")
    axis.set(
        title="Raw Variance of Non-Constant FD001 Sensors",
        xlabel="Sample variance (log scale)",
        ylabel="Sensor",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.text(
        0.01,
        -0.13,
        f"Zero variance: {', '.join(constant)}",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    return _save_figure(figure, path)


def plot_sensor_evolution(
    frame: pd.DataFrame, sensors: list[str], path: Path
) -> Path:
    engine_ids = representative_engine_ids(frame)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, sensor in zip(axes.flat, sensors, strict=True):
        for color, unit_id in zip(COLORS, engine_ids, strict=True):
            engine = frame.loc[frame["unit_id"] == unit_id]
            axis.plot(
                engine["cycle"],
                engine[sensor],
                color=color,
                linewidth=1.0,
                alpha=0.9,
                label=f"Engine {unit_id}",
            )
        axis.set(
            title=sensor.replace("_", " ").title(),
            xlabel="Operational cycle",
            ylabel="Sensor value",
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Selected Sensor Evolution Across Representative Engines", y=1.01)
    figure.tight_layout()
    return _save_figure(figure, path)


def plot_sensor_correlation(frame: pd.DataFrame, path: Path) -> Path:
    nonconstant = [sensor for sensor in SENSOR_COLUMNS if frame[sensor].nunique() > 1]
    correlation = frame[nonconstant].corr()
    figure, axis = plt.subplots(figsize=(10, 8.5))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    axis.set_xticks(range(len(nonconstant)), labels=nonconstant, rotation=55, ha="right")
    axis.set_yticks(range(len(nonconstant)), labels=nonconstant)
    axis.set_title("Correlation Between Non-Constant FD001 Sensors")
    figure.colorbar(image, ax=axis, label="Pearson correlation", shrink=0.82)
    figure.tight_layout()
    return _save_figure(figure, path)


def plot_sensor_trends(summary: pd.DataFrame, path: Path) -> Path:
    plotted = summary.dropna(subset=["median_spearman_rho"]).sort_values("median_spearman_rho")
    lower = plotted["median_spearman_rho"] - plotted["q1_spearman_rho"]
    upper = plotted["q3_spearman_rho"] - plotted["median_spearman_rho"]
    colors = np.where(plotted["median_spearman_rho"] >= 0, COLORS[1], COLORS[0])
    figure, axis = plt.subplots(figsize=(9, 6.5))
    axis.barh(plotted.index, plotted["median_spearman_rho"], color=colors, alpha=0.88)
    axis.errorbar(
        plotted["median_spearman_rho"],
        plotted.index,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#333333",
        capsize=2,
        linewidth=0.8,
    )
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set(
        title="Within-Engine Monotonic Sensor Trends Across FD001",
        xlabel="Median Spearman correlation with cycle (bars); interquartile range (whiskers)",
        ylabel="Sensor",
        xlim=(-1, 1),
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    return _save_figure(figure, path)
