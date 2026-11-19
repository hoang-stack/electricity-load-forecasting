"""Split conformal prediction intervals (Vovk et al., 2005; Lei et al., 2018).

Absolute errors on a held-out calibration period, scaled by each series' level and kept per
forecast horizon, give the half-width that covers a fraction (1 - alpha) of future errors,
provided calibration and test errors are exchangeable. The finite-sample correction uses the
ceil((n + 1)(1 - alpha)) / n quantile.
"""

from __future__ import annotations

import numpy as np


class SplitConformal:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, pred: np.ndarray, true: np.ndarray, scale: np.ndarray) -> SplitConformal:
        """Arrays shaped (origins, series, horizon); ``scale`` is one level per series."""
        r = np.abs(true - pred) / scale[None, :, None]
        r = r.transpose(2, 0, 1).reshape(r.shape[2], -1)  # (horizon, n)
        n = r.shape[1]
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = np.quantile(r, level, axis=1, method="higher")
        return self

    def interval(self, pred: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = self.q[None, None, :] * scale[None, :, None]
        return pred - half, pred + half
