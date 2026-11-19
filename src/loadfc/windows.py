"""Turning a (time x series) load matrix into forecasting problems.

A *forecast origin* t is the hour at which a forecast is issued; the model sees hours
[t - L, t) and predicts hours [t, t + H). Day-ahead forecasting issues one forecast per day at
midnight with H = 24.
"""

from __future__ import annotations

import numpy as np


def daily_origins(n_hours: int, start: int, end: int, lookback: int, horizon: int = 24):
    """Midnight origins t with start <= t and t + horizon <= end, and enough history."""
    first = max(start, lookback)
    first += (-first) % 24  # round up to midnight (hour index 0 is a midnight)
    return np.arange(first, min(end, n_hours) - horizon + 1, 24)


def windows(Y: np.ndarray, origins: np.ndarray, lookback: int, horizon: int):
    """Inputs (n_origins, n_series, lookback) and targets (n_origins, n_series, horizon)."""
    idx_in = origins[:, None] + np.arange(-lookback, 0)[None, :]
    idx_out = origins[:, None] + np.arange(horizon)[None, :]
    X = Y[idx_in].transpose(0, 2, 1)
    T = Y[idx_out].transpose(0, 2, 1)
    return X, T


def day_of_week(origins: np.ndarray, first_weekday: int) -> np.ndarray:
    """Weekday (0 = Monday) of the forecast day, given the weekday of hour 0."""
    return (first_weekday + origins // 24) % 7
