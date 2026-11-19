"""A global linear forecaster in closed form (in the spirit of NLinear, Zeng et al. 2023).

Each window is scaled by its series' training mean and the last observed value is subtracted
(so the model learns *changes*, which transfers across series of very different levels). A
single ridge regression maps the L normalised past hours plus a weekday one-hot to the next
24 hours, shared by all series. Training is one (d x d) solve, accumulated over chunks so the
design matrix never has to fit in memory; the Gram matrix is built once and reused for every
ridge penalty that is tried.
"""

from __future__ import annotations

import numpy as np


def features(
    X: np.ndarray, scale: np.ndarray, weekday: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """X (n_origins, n_series, L) -> design (n_origins * n_series, L + 8) and the offsets."""
    n_o, n_s, L = X.shape
    Xs = X / scale[None, :, None]
    last = Xs[:, :, -1:]
    dow = np.zeros((n_o, n_s, 7), np.float32)
    dow[np.arange(n_o), :, weekday] = 1.0
    D = np.concatenate([Xs - last, dow, np.ones((n_o, n_s, 1), np.float32)], axis=2)
    return D.reshape(n_o * n_s, L + 8), last.reshape(n_o * n_s, 1)


def gram(chunks) -> tuple[np.ndarray, np.ndarray]:
    """``chunks`` yields (design, offset, scaled_target) triples -> (D'D, D'R) in float64."""
    G = b = None
    for D, last, T in chunks:
        D = D.astype(np.float64)
        R = T.astype(np.float64) - last
        G = D.T @ D if G is None else G + D.T @ D
        b = D.T @ R if b is None else b + D.T @ R
    return G, b


class LinearForecaster:
    def __init__(self, lam: float = 1.0):
        self.lam = lam

    @property
    def lookback(self) -> int:
        return self.W.shape[0] - 8  # L past hours + 7 weekday columns + bias

    def solve(self, G: np.ndarray, b: np.ndarray) -> LinearForecaster:
        reg = self.lam * np.eye(G.shape[0])
        reg[-1, -1] = 0.0  # no penalty on the bias
        self.W = np.linalg.solve(G + reg, b).astype(np.float32)
        return self

    def fit_chunks(self, chunks) -> LinearForecaster:
        return self.solve(*gram(chunks))

    def predict(self, X: np.ndarray, scale: np.ndarray, weekday: np.ndarray) -> np.ndarray:
        n_o, n_s, _ = X.shape
        D, last = features(X, scale, weekday)
        out = (D @ self.W + last).reshape(n_o, n_s, -1)
        return out * scale[None, :, None]
