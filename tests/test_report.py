import json

import numpy as np

from loadfc.report import Results, build_report


def fake_outputs(d, n_val=20, n_test=92, n_c=5, seed=0):
    rng = np.random.default_rng(seed)
    scale = rng.uniform(10, 100, n_c).astype(np.float32)

    def truth(n):
        return (scale[None, :, None] * (1 + 0.2 * rng.normal(size=(n, n_c, 24)))).astype(np.float32)

    tv, tt = truth(n_val), truth(n_test)
    arrays = {
        "truth_val": tv,
        "truth_test": tt,
        "scale": scale,
        "train": (scale[None, :] * (1 + 0.2 * rng.normal(size=(2000, n_c)))).astype(np.float32),
        "o_val": 24 * np.arange(n_val),
        "o_test": 24 * (n_val + np.arange(n_test)),
    }
    for name, noise in (
        ("Ensemble (mean, members chosen on validation)", 0.05),
        ("Seasonal naive (week)", 0.3),
    ):
        arrays[f"{name}|val"] = tv + noise * scale[None, :, None] * rng.normal(size=tv.shape)
        arrays[f"{name}|test"] = tt + noise * scale[None, :, None] * rng.normal(size=tt.shape)
    np.savez(d / "forecasts.npz", **arrays)
    (d / "metrics.json").write_text(json.dumps({"clients": n_c, "test_origins": n_test}))
    (d / "clients.json").write_text(
        json.dumps(
            {
                "clients": [f"MT_{i}" for i in range(n_c)],
                "start": "2014-06-01 00:00:00",
                "first_weekday": 6,
            }
        )
    )
    np.savez(d / "lightgbm_quantiles.npz", q10=tt * 0.8, q90=tt * 1.2)


def test_results_rank_models_and_intervals_cover(tmp_path):
    fake_outputs(tmp_path)
    r = Results(tmp_path)
    rows = r.table()
    assert rows[0]["model"].startswith("Ensemble") and rows[0]["nd"] < rows[1]["nd"]
    lo, hi = rows[0]["nd_ci"]
    assert lo <= rows[0]["nd"] <= hi
    gap = r.paired(rows[0]["model"], rows[1]["model"])
    assert gap["diff"] > 0 and gap["ci"][0] > 0
    covers = {(i["model"], i["nominal"]): i["coverage"] for i in r.intervals()}
    assert abs(covers[(rows[0]["model"], 0.8)] - 0.8) < 0.05


def test_build_report_writes_everything(tmp_path):
    fake_outputs(tmp_path)
    md = build_report(tmp_path, tmp_path / "reports")
    assert "| Ensemble" in md.read_text()
    figs = {p.name for p in (tmp_path / "reports" / "figures").iterdir()}
    assert figs == {"nd_by_hour.png", "nd_by_day.png", "week.png"}
