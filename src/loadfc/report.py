"""Figures and reports/results.md from the Kaggle outputs (forecasts.npz, metrics.json)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loadfc import metrics  # noqa: E402
from loadfc.conformal import SplitConformal  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
ACCENT = "#2a78d6"
# Fixed colour per model, never by rank.
STYLE = {
    "Ensemble (mean, members chosen on validation)": "#2a78d6",
    "LightGBM (global, direct)": "#eb6834",
    "N-BEATS (global, GPU)": "#1baf7a",
    "Linear (global ridge)": "#eda100",
    "Seasonal naive (week)": "#e87ba4",
    "Seasonal naive (day)": "#6250d6",
    "Seasonal mean (4 weeks)": MUTED,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# Q4 2014 in Portugal. 1 December was not a holiday from 2013 to 2015 (suspended by law),
# which the data confirms: no error spike that day.
MARKED_DAYS = (
    ("2014-10-26", "clock change"),
    ("2014-12-01", "1 Dec: no holiday"),
    ("2014-12-08", "8 Dec holiday"),
    ("2014-12-25", "Christmas"),
)
DST_DAY = "2014-10-26"  # the dataset stores two hours of load in the 01:00 slot that day


class Results:
    def __init__(self, kaggle_dir: str | Path):
        d = Path(kaggle_dir)
        z = np.load(d / "forecasts.npz")
        self.truth = {"val": z["truth_val"], "test": z["truth_test"]}
        self.scale, self.train = z["scale"], z["train"]
        self.o_test = z["o_test"]
        self.preds: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            if "|" in key:
                model, split = key.split("|")
                self.preds.setdefault(model, {})[split] = z[key]
        self.metrics = json.loads((d / "metrics.json").read_text())
        meta = json.loads((d / "clients.json").read_text())
        self.start = pd.Timestamp(meta["start"])
        q = d / "lightgbm_quantiles.npz"
        self.quantiles = dict(np.load(q)) if q.exists() else None

    def table(self) -> list[dict]:
        y = self.truth["test"]
        rows = []
        for name, p in self.preds.items():
            err, tot = metrics.nd_per_series(p["test"], y)
            rows.append(
                {
                    "model": name,
                    "nd": metrics.nd(p["test"], y),
                    "nd_ci": metrics.bootstrap_nd(err, tot),
                    "nrmse": metrics.nrmse(p["test"], y),
                    "mase": metrics.mase(p["test"], y, self.train),
                    "val_nd": metrics.nd(p["val"], self.truth["val"]),
                }
            )
        return sorted(rows, key=lambda r: r["nd"])

    def paired(self, best: str, other: str, n_boot: int = 2000, seed: int = 0) -> dict:
        """ND(other) - ND(best) with a bootstrap CI over clients (paired)."""
        y = self.truth["test"]
        ea, ta = metrics.nd_per_series(self.preds[best]["test"], y)
        eb, _ = metrics.nd_per_series(self.preds[other]["test"], y)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(ta), (n_boot, len(ta)))
        diff = (eb[idx].sum(1) - ea[idx].sum(1)) / ta[idx].sum(1)
        return {
            "diff": float((eb.sum() - ea.sum()) / ta.sum()),
            "ci": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
        }

    def intervals(self) -> list[dict]:
        rows = []
        for name, p in self.preds.items():
            for alpha in (0.2, 0.05):
                cp = SplitConformal(alpha).fit(p["val"], self.truth["val"], self.scale)
                lo, hi = cp.interval(p["test"], self.scale)
                rows.append(
                    {
                        "model": name,
                        "nominal": 1 - alpha,
                        "coverage": metrics.coverage(lo, hi, self.truth["test"]),
                        "width": metrics.relative_width(lo, hi, self.truth["test"]),
                    }
                )
        if self.quantiles is not None:
            lo, hi = self.quantiles["q10"], self.quantiles["q90"]
            rows.append(
                {
                    "model": "LightGBM quantile regression (10%, 90%)",
                    "nominal": 0.8,
                    "coverage": metrics.coverage(lo, hi, self.truth["test"]),
                    "width": metrics.relative_width(lo, hi, self.truth["test"]),
                }
            )
        return rows

    def nd_without_day(self, name: str, day: str = DST_DAY) -> float:
        """Test ND with one forecast day left out (to size a single-day data artifact)."""
        keep = self.test_dates() != pd.Timestamp(day)
        return metrics.nd(self.preds[name]["test"][keep], self.truth["test"][keep])

    def test_dates(self) -> pd.DatetimeIndex:
        return self.start + pd.to_timedelta(self.o_test, unit="h")


def plot_by_horizon(r: Results, out: Path) -> Path:
    y = r.truth["test"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for name, color in STYLE.items():
        if name not in r.preds:
            continue
        p = r.preds[name]["test"]
        nd_h = np.abs(p - y).sum(axis=(0, 1)) / np.abs(y).sum(axis=(0, 1))
        ax.plot(np.arange(24), 100 * nd_h, color=color, label=name)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlabel("hour of the forecast day")
    ax.set_ylabel("ND (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5)
    ax.set_title("Error by hour of day (test quarter)")
    return _save(fig, out / "nd_by_hour.png")


def plot_by_day(r: Results, best: str, out: Path) -> Path:
    y = r.truth["test"]
    dates = r.test_dates()
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    top = 0.0
    for name in (best, "Seasonal naive (week)"):
        p = r.preds[name]["test"]
        nd_d = np.abs(p - y).sum(axis=(1, 2)) / np.abs(y).sum(axis=(1, 2))
        ax.plot(dates, 100 * nd_d, color=STYLE.get(name, ACCENT), label=name)
        top = max(top, 100 * nd_d.max())
    ax.set_ylim(0, top * 1.45)  # headroom for the day labels
    for day, label in MARKED_DAYS:
        ax.axvline(pd.Timestamp(day), color=BASELINE, lw=0.8, zorder=0)
        ax.annotate(
            label,
            (pd.Timestamp(day), ax.get_ylim()[1]),
            xytext=(-2, -4),
            textcoords="offset points",
            fontsize=7.5,
            color=MUTED,
            rotation=90,
            va="top",
            ha="right",
        )
    ax.set_ylabel("daily ND (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b %Y"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, ncol=2)
    ax.set_title("Error per forecast day: holidays and the clock change are the hard days")
    return _save(fig, out / "nd_by_day.png")


def plot_week(r: Results, best: str, out: Path) -> Path:
    """Total load of all clients over one test week: truth, best forecast, 80% interval."""
    y = r.truth["test"]
    p = r.preds[best]
    cp = SplitConformal(0.2).fit(p["val"], r.truth["val"], r.scale)
    lo, hi = cp.interval(p["test"], r.scale)
    d0 = 28  # four weeks into the quarter
    sl = slice(d0, d0 + 7)
    flat = lambda a: a[sl].sum(axis=1).reshape(-1)  # noqa: E731
    hours = (
        r.start
        + pd.to_timedelta(r.o_test[d0], unit="h")
        + pd.to_timedelta(np.arange(168), unit="h")
    )
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.fill_between(
        hours,
        flat(lo) / 1000,
        flat(hi) / 1000,
        color=ACCENT,
        alpha=0.15,
        lw=0,
        label="80% conformal interval (sum of per-client bounds)",
    )
    ax.plot(hours, flat(y) / 1000, color=INK, lw=1.4, label="actual")
    ax.plot(hours, flat(p["test"]) / 1000, color=ACCENT, lw=1.4, label=best)
    ax.set_ylabel("total load (MW)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    ax.set_title("One test week, all clients summed")
    fig.autofmt_xdate()
    return _save(fig, out / "week.png")


def build_report(kaggle_dir="kaggle/output/main", report_dir="reports") -> Path:
    r = Results(kaggle_dir)
    out = Path(report_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _style()
    rows = r.table()
    best = rows[0]["model"]
    plot_by_horizon(r, out / "figures")
    plot_by_day(r, best, out / "figures")
    plot_week(r, best, out / "figures")
    paired = {row["model"]: r.paired(best, row["model"]) for row in rows[1:]}
    intervals = r.intervals()
    dst = {
        row["model"]: {"nd": row["nd"], "nd_without_dst_day": r.nd_without_day(row["model"])}
        for row in rows
    }
    (out / "summary.json").write_text(
        json.dumps(
            {
                "models": rows,
                "paired_vs_best": paired,
                "intervals": intervals,
                "dst_day_effect": dst,
                "kaggle": r.metrics,
            },
            indent=1,
        )
    )
    lines = [
        f"# Results (test: Q4 2014, {r.metrics['clients']} clients, "
        f"{r.metrics['test_origins']} daily forecasts each)",
        "",
        "| Model | ND (95% CI) | NRMSE | MASE | Validation ND | ND gap to best (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lo, hi = row["nd_ci"]
        gap = paired.get(row["model"])
        gap_s = (
            (f"+{100 * gap['diff']:.2f} pp ({100 * gap['ci'][0]:.2f} to {100 * gap['ci'][1]:.2f})")
            if gap
            else "–"
        )
        lines.append(
            f"| {row['model']} | {100 * row['nd']:.2f}% ({100 * lo:.2f}–{100 * hi:.2f})"
            f" | {row['nrmse']:.3f} | {row['mase']:.3f} | {100 * row['val_nd']:.2f}% | "
            f"{gap_s} |"
        )
    lines += [
        "",
        "## Prediction intervals (calibrated on the validation quarter)",
        "",
        "| Method | Nominal | Coverage on test | Width / mean load |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {i['model']} | {100 * i['nominal']:.0f}% | {100 * i['coverage']:.1f}% | "
        f"{i['width']:.3f} |"
        for i in intervals
    ]
    md = out / "results.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md
