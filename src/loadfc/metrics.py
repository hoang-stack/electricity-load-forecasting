"""Forecast accuracy and interval quality, aggregated over series, origins and horizons.

* ND (normalised deviation) = sum |y_hat - y| / sum |y|, the scale-free error used for this
  dataset since DeepAR (Salinas et al., 2020).
* NRMSE = sqrt(mean (y_hat - y)^2) / mean |y|.
* MASE (Hyndman & Koehler, 2006): MAE divided by the in-sample MAE of the weekly seasonal
  naive forecast, per series, then averaged. Below 1 = better than "same hour last week".
"""

from __future__ import annotations

import numpy as np


def nd(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).sum() / np.abs(true).sum())


def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)) / np.mean(np.abs(true)))


def nd_per_series(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sum |error|, sum |y|) per series, inputs shaped (origins, series, horizon)."""
    return np.abs(pred - true).sum(axis=(0, 2)), np.abs(true).sum(axis=(0, 2))


def mase(pred: np.ndarray, true: np.ndarray, train: np.ndarray, period: int = 168) -> float:
    """``train`` is (time, series): the scale is each series' mean |y_t - y_{t-period}|."""
    scale = np.abs(train[period:] - train[:-period]).mean(axis=0)
    mae = np.abs(pred - true).mean(axis=(0, 2))
    ok = scale > 0
    return float((mae[ok] / scale[ok]).mean())


def bootstrap_nd(abs_err: np.ndarray, abs_y: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% CI of ND resampling *series* (the independent units)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(abs_err), (n_boot, len(abs_err)))
    vals = abs_err[idx].sum(axis=1) / abs_y[idx].sum(axis=1)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def coverage(lo: np.ndarray, hi: np.ndarray, true: np.ndarray) -> float:
    return float(((true >= lo) & (true <= hi)).mean())


def relative_width(lo: np.ndarray, hi: np.ndarray, true: np.ndarray) -> float:
    """Mean interval width divided by mean |y| (comparable across models)."""
    return float((hi - lo).mean() / np.abs(true).mean())
