"""Per-subject z-score normalization for ECoG features.

All ``fit_*`` functions compute statistics from training data only.
The matching ``apply_*`` functions normalize any split using those statistics,
preventing data leakage from validation or test sets.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


def _safe_std(std: np.ndarray, eps: float) -> np.ndarray:
    """Replace near-zero standard deviations with 1.0 to avoid division by zero."""
    std = std.copy()
    std[std < eps] = 1.0
    return std


@dataclass
class ZScoreStats:
    """Stores mean and std arrays for z-score normalization."""
    mean: np.ndarray
    std: np.ndarray


def fit_zscore_2d(x: np.ndarray, eps: float = 1e-6) -> ZScoreStats:
    """Fit z-score stats for a 2-D array of shape (N, D).

    Statistics are computed over the N axis (one scalar per feature dimension).
    """
    mean = x.mean(axis=0)
    std = _safe_std(x.std(axis=0), eps)
    return ZScoreStats(mean=mean, std=std)


def apply_zscore_2d(x: np.ndarray, st: ZScoreStats) -> np.ndarray:
    """Apply 2-D z-score normalization using pre-fitted stats."""
    return (x - st.mean[None, :]) / st.std[None, :]


def fit_zscore_3d_per_channel(x: np.ndarray, eps: float = 1e-6) -> ZScoreStats:
    """Fit per-channel z-score stats for a 3-D array of shape (N, C, T).

    Mean and std are computed over the (N, T) axes, yielding one value per
    channel C.
    """
    mean = x.mean(axis=(0, 2))   # [C]
    std = _safe_std(x.std(axis=(0, 2)), eps)  # [C]
    return ZScoreStats(mean=mean, std=std)


def apply_zscore_3d_per_channel(x: np.ndarray, st: ZScoreStats) -> np.ndarray:
    """Apply per-channel z-score normalization to a (N, C, T) array."""
    return (x - st.mean[None, :, None]) / st.std[None, :, None]
