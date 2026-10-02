"""RUL regression metrics. The error is d = predicted - true, so d > 0 is a late prediction."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(errors**2)))


def mae(y_true, y_pred) -> float:
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(errors)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score (Saxena et al., 2008), summed over engines; lower is better.

    Early predictions cost exp(-d/13) - 1, late ones exp(d/10) - 1. The paper prints
    a1=10, a2=13 under its equation, which would punish early predictions harder; its
    text says late predictions must cost more, which is the convention used in practice.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(penalties.sum())
