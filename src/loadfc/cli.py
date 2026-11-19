"""Command line: ``loadfc {report,forecast}``."""

from __future__ import annotations

import argparse

import numpy as np


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="loadfc", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="figures + reports/results.md from the Kaggle outputs")
    rp.add_argument("--kaggle-dir", default="kaggle/output/main")
    rp.add_argument("--report-dir", default="reports")
    fc = sub.add_parser("forecast", help="next 24 hours from the last 168 hourly loads (CSV)")
    fc.add_argument("history", help="file with 168 numbers, one per line, oldest first")
    fc.add_argument(
        "--weekday", type=int, required=True, help="weekday of the forecast day (0=Mon)"
    )
    fc.add_argument("--scale", type=float, default=None, help="series level (default: mean)")
    fc.add_argument("--weights", default="artifacts/linear_weights.npy")
    args = p.parse_args(argv)
    if args.cmd == "report":
        from loadfc.report import build_report

        print(f"report written to {build_report(args.kaggle_dir, args.report_dir)}")
    else:
        from loadfc.linear import LinearForecaster

        hist = np.loadtxt(args.history, dtype=np.float32).reshape(1, 1, -1)
        if hist.shape[-1] != 168:
            raise SystemExit("need exactly 168 hourly values")
        model = LinearForecaster()
        model.W = np.load(args.weights)
        scale = np.array([args.scale or float(hist.mean())], np.float32)
        pred = model.predict(hist, scale, np.array([args.weekday]))[0, 0]
        for h, v in enumerate(pred):
            print(f"{h:02d}:00  {v:.2f}")


if __name__ == "__main__":
    main()
