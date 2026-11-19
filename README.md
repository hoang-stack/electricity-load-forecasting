# electricity-load-forecasting

[![CI](https://github.com/hoang-stack/electricity-load-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/hoang-stack/electricity-load-forecasting/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Day-ahead forecasts of hourly electricity load for hundreds of clients, with honest
uncertainty.** Every midnight, each model predicts the next 24 hours for every client of the
[UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
dataset (Portugal, CC BY 4.0). The comparison covers seasonal baselines, a global linear model
solved in closed form, a global LightGBM, and N-BEATS on a GPU, and every model gets conformal
prediction intervals whose coverage is checked on unseen data.

It also tests a claim that has been debated since Zeng et al. (2023): on standard benchmarks,
deep forecasting models are often not better than a well-built linear model.

<!-- RESULTS -->

## Protocol

- **Data:** 15-minute readings in kW, averaged to hourly, 2012–2014. Clients with more than 5%
  zero hours are dropped (they connected late or were switched off for long periods).
- **Task:** at each midnight, forecast hours 00:00–23:00 of that day from the data available
  up to that midnight. Models may look back up to two weeks (336 hours).
- **Splits in time:** train until 30 June 2014, **validation** Q3 2014 (every choice:
  regularisation, early stopping, conformal calibration), **test** Q4 2014 (92 forecast days per
  client), scored once.
- **Metrics:** ND = Σ|error| / Σ|load| (the standard for this dataset since DeepAR), NRMSE and
  MASE against the weekly seasonal naive forecast. 95% CIs bootstrap *clients*. Model gaps use
  a paired bootstrap over clients.
- **Intervals:** split conformal prediction, using validation errors scaled by each client's
  level, separately per forecast hour. Compared with LightGBM quantile regression.

## Models

| Model | Idea |
| --- | --- |
| Seasonal naive (day / week) | Tomorrow = today, or the same day last week |
| Seasonal mean | Average of the same hour on the same weekday over the last 4 weeks |
| **Linear (global ridge)** ([linear.py](src/loadfc/linear.py)) | NLinear-style: scale by client level, subtract the last value, one ridge regression from the past 1 or 2 weeks + weekday to the next 24 hours, shared by all clients, solved in closed form (lookback and penalty chosen on validation) |
| **LightGBM (global, direct)** | One model for all clients and horizons: recent hours, same hour 1–2 days and 1–2 weeks ago, rolling statistics, calendar, client id. L1 objective |
| **N-BEATS** (Oreshkin et al., 2020) | Generic architecture: 6 residual MLP blocks with backcast/forecast over the past week, per-window level normalisation, trained on a T4 GPU, early-stopped on validation |
| Ensemble | Plain mean of the subset of {linear, LightGBM, N-BEATS} with the lowest validation ND |

<!-- ANALYSIS -->

## Quickstart

```bash
git clone https://github.com/hoang-stack/electricity-load-forecasting.git
cd electricity-load-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # unit tests on synthetic series
```

The experiment runs as a Kaggle notebook ([kaggle/forecast.py](kaggle/forecast.py)). It
downloads the data from UCI itself, and needs a dataset containing `src/loadfc` plus a T4 GPU.
Copy `kaggle/kernel-metadata.example.json` to `kernel-metadata.json`, set your username, and run
`kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4`. Then download `forecasts.npz`,
`metrics.json`, `clients.json` and `lightgbm_quantiles.npz` to `kaggle/output/main/`, and run
`loadfc report`.

The global linear model is a 176 × 24 matrix. `loadfc forecast history.csv --weekday 0`
(with `linear_weights.npy` in `artifacts/`) forecasts any series from its last 168 hours,
without PyTorch or a GPU.

## Project layout

```text
src/loadfc/
├── windows.py      forecast origins and (input, target) windows from a time x series matrix
├── baselines.py    seasonal naive and seasonal mean
├── linear.py       global NLinear-style ridge regression, solved by streaming normal equations
├── metrics.py      ND, NRMSE, MASE, client bootstrap, interval coverage and width
├── conformal.py    split conformal intervals per horizon with finite-sample correction
├── report.py       tables and figures from the Kaggle outputs
└── cli.py          `loadfc report | forecast`
kaggle/forecast.py  data download, all models (LightGBM, PyTorch N-BEATS on GPU), intervals
```

<!-- LIMITATIONS -->

## References

- Trindade, A. (2015). ElectricityLoadDiagrams20112014. UCI Machine Learning Repository.
- Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). Are Transformers effective for time series forecasting? *AAAI.*
- Oreshkin, B. et al. (2020). N-BEATS: neural basis expansion analysis for interpretable time series forecasting. *ICLR.*
- Salinas, D. et al. (2020). DeepAR: probabilistic forecasting with autoregressive recurrent networks. *IJF.*
- Hyndman, R., Koehler, A. (2006). Another look at measures of forecast accuracy. *IJF.*
- Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS.*
- Lei, J. et al. (2018). Distribution-free predictive inference for regression. *JASA.*
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). M5 accuracy competition: results, findings, and conclusions. *IJF.*
