# Time Series Forecasting

[![CI](https://github.com/cerenaaa/time-series-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/cerenaaa/time-series-forecasting/actions)

Revenue and demand forecasting pipeline: ARIMA, Exponential Smoothing, Prophet-style decomposition, and an ensemble combiner. Built for business forecasting with uncertainty quantification and backtesting.

## Models

| Model | Strengths | Best for |
|---|---|---|
| SARIMA | Handles seasonality + autocorrelation | Regular seasonal patterns |
| ETS | Multiplicative trend + seasonality | Exponential growth series |
| Decomposition | Interpretable trend/seasonal/residual | Exploratory analysis |
| Ensemble | Reduces model risk | Production forecasting |

## Structure
```
time-series-forecasting/
├── data/
│   └── synthetic_revenue.py      # Synthetic revenue time series generator
├── models/
│   ├── sarima.py                 # SARIMA with auto-order selection
│   ├── ets.py                    # Exponential Smoothing (ETS)
│   ├── decomposition.py          # STL decomposition
│   └── ensemble.py               # Weighted ensemble combiner
├── evaluation/
│   └── backtester.py             # Walk-forward backtesting + metrics
└── forecast.py
```

## Quickstart
```bash
pip install -r requirements.txt
python forecast.py --horizon 12 --model ensemble
```
