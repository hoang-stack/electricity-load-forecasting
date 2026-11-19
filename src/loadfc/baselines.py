"""Seasonal baselines: the bar every forecasting model has to clear."""

from __future__ import annotations

import numpy as np


def seasonal_naive(Y: np.ndarray, origins: np.ndarray, period: int, horizon: int = 24):
    """y_hat[t + h] = y[t + h - period] (period >= horizon), shape (n_origins, n_series, H)."""
    if period < horizon:
        raise ValueError("period must be at least the horizon")
    idx = origins[:, None] + np.arange(horizon)[None, :] - period
    return Y[idx].transpose(0, 2, 1)


def seasonal_mean(Y: np.ndarray, origins: np.ndarray, weeks: int = 4, horizon: int = 24):
    """Average of the same hour on the same weekday over the last ``weeks`` weeks."""
    return np.mean(
        [seasonal_naive(Y, origins, 168 * k, horizon) for k in range(1, weeks + 1)], axis=0
    )
