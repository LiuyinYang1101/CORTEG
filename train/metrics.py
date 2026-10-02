"""Regression metrics for ECoG finger trajectory decoding."""
from __future__ import annotations
import numpy as np


def corr_per_dim(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Return per-dimension Pearson correlation between predictions and targets.

    Parameters
    ----------
    y_pred, y_true : np.ndarray of shape (N, D)

    Returns
    -------
    np.ndarray of shape (D,) — NaN where a dimension has zero variance.
    """
    assert y_pred.shape == y_true.shape and y_pred.ndim == 2
    D = y_true.shape[1]
    out = np.full((D,), np.nan, dtype=np.float64)
    for d in range(D):
        a = y_pred[:, d]
        b = y_true[:, d]
        if np.std(a) == 0 or np.std(b) == 0:
            out[d] = np.nan
        else:
            out[d] = np.corrcoef(a, b)[0, 1]
    return out


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error across all elements."""
    return float(np.mean((y_pred - y_true) ** 2))


def nanmean(x: np.ndarray) -> float:
    """Mean of array ignoring NaN values."""
    return float(np.nanmean(x))
