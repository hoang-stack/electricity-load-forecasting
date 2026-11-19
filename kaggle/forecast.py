"""Day-ahead load forecasting for ~320 Portuguese clients (UCI ElectricityLoadDiagrams20112014).

Every model forecasts the 24 hours of each day at midnight, using only data up to that
midnight. Choices are made on the validation quarter (Q3 2014); the test quarter (Q4 2014) is
scored once. Outputs (small): per-model forecasts for validation and test, the truth, per-client
scales, metrics.json. Point metrics and conformal intervals use the project's own `loadfc` code.
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SMOKE = int(os.environ.get("SMOKE", 0))  # >0: only this many clients (pipeline test)
OUT = Path("/kaggle/working")
URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


sys.path.insert(
    0, str(Path(glob.glob("/kaggle/input/**/loadfc/metrics.py", recursive=True)[0]).parent.parent)
)
from loadfc import baselines, linear, metrics, windows  # noqa: E402
from loadfc.conformal import SplitConformal  # noqa: E402

# --- data -----------------------------------------------------------------------------------
with urllib.request.urlopen(URL, timeout=600) as r:
    payload = r.read()
with zipfile.ZipFile(io.BytesIO(payload)) as zf:
    name = next(n for n in zf.namelist() if n.endswith(".txt"))
    raw = pd.read_csv(zf.open(name), sep=";", decimal=",", index_col=0, parse_dates=True)
log("raw", raw.shape)
hourly = raw.loc["2012-01-01":"2014-12-31"].resample("1h").mean()  # kW, averaged per hour
zero_share = (hourly == 0).mean()
keep = zero_share[zero_share < 0.05].index  # drop clients connected late or often off
hourly = hourly[keep]
if SMOKE:
    hourly = hourly.iloc[:, :SMOKE]
Y = hourly.to_numpy(np.float32)
clients = list(hourly.columns)
first_weekday = hourly.index[0].weekday()
T_VAL, T_TEST = hourly.index.get_loc("2014-07-01 00:00"), hourly.index.get_loc("2014-10-01 00:00")
log(f"{len(clients)} clients x {len(Y)} hours; val from hour {T_VAL}, test from {T_TEST}")
scale = Y[:T_VAL].mean(axis=0)
L, H = 168, 24
o_train = windows.daily_origins(len(Y), 0, T_VAL, L)
o_val = windows.daily_origins(len(Y), T_VAL, T_TEST, L)
o_test = windows.daily_origins(len(Y), T_TEST, len(Y), L)
_, truth_val = windows.windows(Y, o_val, L, H)
_, truth_test = windows.windows(Y, o_test, L, H)
preds: dict[str, dict[str, np.ndarray]] = {}
info: dict = {
    "clients": len(clients),
    "hours": len(Y),
    "train_origins": len(o_train),
    "val_origins": len(o_val),
    "test_origins": len(o_test),
}


def add(name: str, val: np.ndarray, test: np.ndarray, **extra) -> None:
    preds[name] = {"val": val.astype(np.float32), "test": test.astype(np.float32)}
    info.setdefault("models", {})[name] = {
        "val_nd": metrics.nd(val, truth_val),
        "test_nd": metrics.nd(test, truth_test),
        **extra,
    }
    log(
        f"{name:28s} val ND {metrics.nd(val, truth_val):.4f}  "
        f"test ND {metrics.nd(test, truth_test):.4f}"
    )


# --- baselines --------------------------------------------------------------------------------
for name, fn in (
    ("Seasonal naive (day)", lambda o: baselines.seasonal_naive(Y, o, 24)),
    ("Seasonal naive (week)", lambda o: baselines.seasonal_naive(Y, o, 168)),
    ("Seasonal mean (4 weeks)", lambda o: baselines.seasonal_mean(Y, o, 4)),
):
    add(name, fn(o_val), fn(o_test))


# --- global linear model (closed form) --------------------------------------------------------
def lin_chunks(origins, size=64):
    for s in range(0, len(origins), size):
        o = origins[s : s + size]
        X, T = windows.windows(Y, o, L, H)
        D, last = linear.features(X, scale, windows.day_of_week(o, first_weekday))
        yield D, last, (T / scale[None, :, None]).reshape(-1, H)


best = None
for lam in (0.1, 1.0, 10.0, 100.0, 1000.0):
    m = linear.LinearForecaster(lam).fit_chunks(lin_chunks(o_train))
    Xv, _ = windows.windows(Y, o_val, L, H)
    nd = metrics.nd(m.predict(Xv, scale, windows.day_of_week(o_val, first_weekday)), truth_val)
    log(f"linear lambda={lam}: val ND {nd:.4f}")
    if best is None or nd < best[0]:
        best = (nd, lam, m)
_, lam, lin = best
Xv, _ = windows.windows(Y, o_val, L, H)
Xt, _ = windows.windows(Y, o_test, L, H)
add(
    "Linear (global ridge)",
    lin.predict(Xv, scale, windows.day_of_week(o_val, first_weekday)),
    lin.predict(Xt, scale, windows.day_of_week(o_test, first_weekday)),
    lam=lam,
)
np.save(OUT / "linear_weights.npy", lin.W)

# --- LightGBM, one global direct model ---------------------------------------------------------
try:
    import lightgbm as lgb
except ImportError:  # not in every Kaggle image
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm"], check=True)
    import lightgbm as lgb


def gbm_table(origins):
    """One row per (origin, client, horizon); every feature is scaled by the client's level."""
    n_o, n_c = len(origins), Y.shape[1]
    Ys = Y / scale[None, :]
    X, _ = windows.windows(Ys, origins, L, H)  # (o, c, 168)
    o = np.repeat(origins, n_c * H)
    c = np.tile(np.repeat(np.arange(n_c), H), n_o)
    h = np.tile(np.arange(H), n_o * n_c)
    t = o + h
    cols = {
        "client": c,
        "h": h,
        "dow": (first_weekday + t // 24) % 7,
        "month": hourly.index.month.to_numpy()[np.minimum(t, len(Y) - 1)],
        "lag_d1": Ys[t - 24, c],
        "lag_d2": Ys[t - 48, c],
        "lag_w1": Ys[t - 168, c],
        "lag_w2": Ys[t - 336, c],  # every origin used has >= 336 hours of history
        "last": np.repeat(X[:, :, -1].reshape(-1), H),
        "mean_24": np.repeat(X[:, :, -24:].mean(2).reshape(-1), H),
        "mean_168": np.repeat(X.mean(2).reshape(-1), H),
        "std_168": np.repeat(X.std(2).reshape(-1), H),
    }
    for k in range(1, 25):  # the last day, hour by hour
        cols[f"prev_{k}"] = np.repeat(X[:, :, -k].reshape(-1), H)
    df = pd.DataFrame(cols)
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype(np.float32)  # ~7M training rows: halve the memory
    df["client"] = df["client"].astype("category")
    return df, Ys[t, c]


tr_origins = o_train[o_train >= 336]
Xtr, ytr = gbm_table(tr_origins)
Xva, yva = gbm_table(o_val)  # val/test origins are >= 336 too (they start in July 2014)
Xte, _ = gbm_table(o_test)
log(f"LightGBM rows: train {len(Xtr):,}")
params = {
    "objective": "l1",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "verbose": -1,
    "num_threads": os.cpu_count(),
}
dtr = lgb.Dataset(Xtr, ytr, categorical_feature=["client"], free_raw_data=False)
dva = lgb.Dataset(Xva, yva, reference=dtr)
gbm = lgb.train(
    params,
    dtr,
    num_boost_round=200 if SMOKE else 2000,
    valid_sets=[dva],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
shape_v, shape_t = (len(o_val), Y.shape[1], H), (len(o_test), Y.shape[1], H)
add(
    "LightGBM (global, direct)",
    gbm.predict(Xva).reshape(shape_v) * scale[None, :, None],
    gbm.predict(Xte).reshape(shape_t) * scale[None, :, None],
    trees=gbm.best_iteration,
)
imp = dict(zip(gbm.feature_name(), gbm.feature_importance("gain").tolist(), strict=True))
info["lightgbm_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])
quantiles = {}
for q in (0.1, 0.9):
    p = {**params, "objective": "quantile", "alpha": q}
    m = lgb.train(p, dtr, num_boost_round=gbm.best_iteration)
    quantiles[q] = m.predict(Xte).reshape(shape_t) * scale[None, :, None]
np.savez_compressed(
    OUT / "lightgbm_quantiles.npz",
    q10=quantiles[0.1].astype(np.float32),
    q90=quantiles[0.9].astype(np.float32),
)
del Xtr, dtr
log("LightGBM done")

# --- N-BEATS (generic), a global deep model on the GPU -----------------------------------------
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, width=512, layers=4):
        super().__init__()
        mods, d = [], L + 7
        for _ in range(layers):
            mods += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.mlp = nn.Sequential(*mods)
        self.back = nn.Linear(width, L)
        self.fore = nn.Linear(width, H)

    def forward(self, x, dow):
        z = self.mlp(torch.cat([x, dow], dim=1))
        return self.back(z), self.fore(z)


class NBeats(nn.Module):
    """Oreshkin et al. (ICLR 2020), generic architecture: residual stacking of MLP blocks, each
    removing what it explains from the input (backcast) and adding to the forecast."""

    def __init__(self, blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(blocks)])

    def forward(self, x, dow):
        f = 0
        for b in self.blocks:
            back, fore = b(x, dow)
            x = x - back
            f = f + fore
        return f


def batch(origins, clients):
    X = Y[origins[:, None] + np.arange(-L, 0)[None, :], clients[:, None]]
    T = Y[origins[:, None] + np.arange(H)[None, :], clients[:, None]]
    mu = X.mean(1, keepdims=True) + 1e-3  # per-window level (RevIN-style)
    dow = np.eye(7, dtype=np.float32)[(first_weekday + origins // 24) % 7]
    return (
        torch.from_numpy(X / mu).to(device),
        torch.from_numpy(dow).to(device),
        torch.from_numpy(T / mu).to(device),
        mu,
    )


@torch.no_grad()
def nbeats_forecast(model, origins):
    model.eval()
    n_c = Y.shape[1]
    o = np.repeat(origins, n_c)
    c = np.tile(np.arange(n_c), len(origins))
    out = []
    for s in range(0, len(o), 8192):
        x, dow, _, mu = batch(o[s : s + 8192], c[s : s + 8192])
        out.append(model(x, dow).cpu().numpy() * mu)
    return np.concatenate(out).reshape(len(origins), n_c, H)


model = NBeats().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
rng = np.random.default_rng(0)
all_hours = np.arange(L, T_VAL - H + 1)  # any hour can be an origin during training
best_nd, best_state, history = np.inf, None, []
for epoch in range(3 if SMOKE else 30):
    model.train()
    for _ in range(200):
        o = rng.choice(all_hours, 1024)
        c = rng.integers(0, Y.shape[1], 1024)
        x, dow, t, _ = batch(o, c)
        loss = (model(x, dow) - t).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    nd_val = metrics.nd(nbeats_forecast(model, o_val), truth_val)
    history.append({"epoch": epoch + 1, "val_nd": nd_val})
    if nd_val < best_nd:
        best_nd, best_state = nd_val, {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
add(
    "N-BEATS (global, GPU)",
    nbeats_forecast(model, o_val),
    nbeats_forecast(model, o_test),
    history=history,
    best_epoch=int(np.argmin([h["val_nd"] for h in history]) + 1),
)

# --- ensemble and conformal intervals ---------------------------------------------------------
add(
    "Ensemble (linear + LightGBM + N-BEATS)",
    np.mean(
        [
            preds[k]["val"]
            for k in ("Linear (global ridge)", "LightGBM (global, direct)", "N-BEATS (global, GPU)")
        ],
        axis=0,
    ),
    np.mean(
        [
            preds[k]["test"]
            for k in ("Linear (global ridge)", "LightGBM (global, direct)", "N-BEATS (global, GPU)")
        ],
        axis=0,
    ),
)
intervals = {}
for name, p in preds.items():
    for alpha in (0.2, 0.05):
        cp = SplitConformal(alpha).fit(p["val"], truth_val, scale)
        lo, hi = cp.interval(p["test"], scale)
        intervals[f"{name}|{1 - alpha:.2f}"] = {
            "coverage": metrics.coverage(lo, hi, truth_test),
            "width": metrics.relative_width(lo, hi, truth_test),
        }
lo, hi = quantiles[0.1], quantiles[0.9]
intervals["LightGBM quantile regression|0.80"] = {
    "coverage": metrics.coverage(lo, hi, truth_test),
    "width": metrics.relative_width(lo, hi, truth_test),
}
info["intervals"] = intervals

np.savez_compressed(
    OUT / "forecasts.npz",
    truth_val=truth_val,
    truth_test=truth_test,
    scale=scale,
    train=Y[:T_VAL],
    o_val=o_val,
    o_test=o_test,
    **{f"{k}|{s}": v[s] for k, v in preds.items() for s in ("val", "test")},
)
(OUT / "clients.json").write_text(
    json.dumps({"clients": clients, "start": str(hourly.index[0]), "first_weekday": first_weekday})
)
(OUT / "metrics.json").write_text(json.dumps(info, indent=2))
log("done")
