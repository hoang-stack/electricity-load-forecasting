import numpy as np
import pytest

from loadfc import baselines, conformal, linear, metrics, windows


def toy(n_hours=24 * 7 * 30, n_series=6, seed=0):
    """Daily and weekly seasonality at different levels per series, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    base = 10 + 3 * np.sin(2 * np.pi * t / 24) + 2 * (((t // 24) % 7) < 5)
    levels = rng.uniform(1, 50, n_series)
    return base[:, None] * levels[None, :] * (1 + 0.05 * rng.normal(size=(n_hours, n_series)))


def test_origins_are_midnights_with_history():
    o = windows.daily_origins(24 * 40, start=24 * 10 + 5, end=24 * 20, lookback=168)
    assert (o % 24 == 0).all() and o[0] == 24 * 11 and o[-1] + 24 <= 24 * 20


def test_windows_align_inputs_and_targets():
    Y = np.arange(500, dtype=float)[:, None] * np.array([1.0, 2.0])
    X, T = windows.windows(Y, np.array([200]), lookback=3, horizon=2)
    np.testing.assert_array_equal(X[0, 0], [197, 198, 199])
    np.testing.assert_array_equal(T[0, 1], [400, 402])


def test_seasonal_naive_and_mean():
    Y = toy()
    o = np.array([24 * 100])
    np.testing.assert_array_equal(baselines.seasonal_naive(Y, o, 24)[0, :, 0], Y[o[0] - 24])
    with pytest.raises(ValueError):
        baselines.seasonal_naive(Y, o, 12)
    assert baselines.seasonal_mean(Y, o).shape == (1, Y.shape[1], 24)


def test_linear_forecaster_beats_seasonal_naive_on_seasonal_data():
    Y = toy().astype(np.float32)
    scale = Y[: 24 * 140].mean(axis=0)
    tr = windows.daily_origins(len(Y), 0, 24 * 140, 168)
    te = windows.daily_origins(len(Y), 24 * 150, len(Y), 168)

    def chunks():
        X, T = windows.windows(Y, tr, 168, 24)
        D, last = linear.features(X, scale, windows.day_of_week(tr, 0))
        yield D, last, (T / scale[None, :, None]).reshape(-1, 24)

    model = linear.LinearForecaster(lam=1.0).fit_chunks(chunks())
    X, T = windows.windows(Y, te, 168, 24)
    pred = model.predict(X, scale, windows.day_of_week(te, 0))
    naive = baselines.seasonal_naive(Y, te, 24)
    assert metrics.nd(pred, T) < metrics.nd(naive, T)


def test_metrics():
    true = np.ones((2, 3, 4))
    pred = true * 1.1
    assert metrics.nd(pred, true) == pytest.approx(0.1)
    assert metrics.nrmse(pred, true) == pytest.approx(0.1)
    err, tot = metrics.nd_per_series(pred, true)
    assert err.shape == (3,) and tot.sum() == 24
    lo, hi = metrics.bootstrap_nd(err, tot)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    train = np.tile(np.array([1.0, 3.0]), 200)[:, None] * np.ones((1, 3))
    assert metrics.mase(pred, true, train, period=1) == pytest.approx(0.1 / 2)


def test_split_conformal_reaches_nominal_coverage():
    rng = np.random.default_rng(0)
    scale = np.array([1.0, 10.0, 100.0])
    true_cal = rng.normal(0, 1, (300, 3, 24)) * scale[None, :, None]
    true_te = rng.normal(0, 1, (300, 3, 24)) * scale[None, :, None]
    zero = np.zeros_like(true_cal)
    cp = conformal.SplitConformal(alpha=0.1).fit(zero, true_cal, scale)
    lo, hi = cp.interval(zero, scale)
    assert metrics.coverage(lo, hi, true_te) == pytest.approx(0.9, abs=0.02)
