# Results (test: Q4 2014, 314 clients, 92 daily forecasts each)

| Model | ND (95% CI) | NRMSE | MASE | Validation ND | ND gap to best (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM (global, direct) | 6.31% (5.69–6.85) | 0.566 | 0.686 | 5.60% | – |
| Ensemble (mean, members chosen on validation) | 6.34% (5.68–6.91) | 0.567 | 0.677 | 5.57% | +0.04 pp (-0.06 to 0.10) |
| N-BEATS (global, GPU) | 6.79% (6.05–7.40) | 0.594 | 0.714 | 5.92% | +0.48 pp (0.28 to 0.62) |
| Linear (global ridge) | 7.77% (6.90–8.53) | 0.639 | 0.825 | 7.03% | +1.46 pp (1.17 to 1.71) |
| Seasonal naive (day) | 8.47% (7.85–9.00) | 0.671 | 0.999 | 6.92% | +2.16 pp (1.85 to 2.67) |
| Seasonal naive (week) | 9.60% (7.71–11.48) | 1.058 | 0.957 | 9.17% | +3.29 pp (1.86 to 4.68) |
| Seasonal mean (4 weeks) | 10.60% (8.18–13.19) | 1.164 | 0.991 | 9.18% | +4.29 pp (2.30 to 6.39) |

## Prediction intervals (calibrated on the validation quarter)

| Method | Nominal | Coverage on test | Width / mean load |
| --- | ---: | ---: | ---: |
| Seasonal naive (day) | 80% | 78.7% | 0.284 |
| Seasonal naive (day) | 95% | 94.0% | 0.751 |
| Seasonal naive (week) | 80% | 81.5% | 0.326 |
| Seasonal naive (week) | 95% | 95.9% | 0.789 |
| Seasonal mean (4 weeks) | 80% | 79.3% | 0.312 |
| Seasonal mean (4 weeks) | 95% | 95.6% | 0.755 |
| Linear (global ridge) | 80% | 79.9% | 0.265 |
| Linear (global ridge) | 95% | 94.8% | 0.602 |
| LightGBM (global, direct) | 80% | 80.0% | 0.214 |
| LightGBM (global, direct) | 95% | 94.9% | 0.492 |
| N-BEATS (global, GPU) | 80% | 79.7% | 0.222 |
| N-BEATS (global, GPU) | 95% | 94.8% | 0.500 |
| Ensemble (mean, members chosen on validation) | 80% | 79.8% | 0.211 |
| Ensemble (mean, members chosen on validation) | 95% | 94.8% | 0.479 |
| LightGBM quantile regression (10%, 90%) | 80% | 75.7% | 0.167 |
