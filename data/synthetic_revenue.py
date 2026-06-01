"""Synthetic revenue time series with trend, seasonality, and noise."""
import numpy as np
import pandas as pd


def generate_revenue_series(n_months: int = 60, seed: int = 42,
                              trend_slope: float = 500.0,
                              seasonal_amplitude: float = 2000.0,
                              noise_std: float = 800.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    dates = pd.date_range("2019-01-01", periods=n_months, freq="MS")

    trend = 10_000 + trend_slope * t
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 12 - np.pi / 2)
    holiday_boost = np.zeros(n_months)
    for i, d in enumerate(dates):
        if d.month == 12:
            holiday_boost[i] = 3000
        elif d.month == 11:
            holiday_boost[i] = 1500

    noise = rng.normal(0, noise_std, n_months)
    revenue = (trend + seasonal + holiday_boost + noise).clip(0)

    df = pd.DataFrame({"date": dates, "revenue": revenue.round(2),
                        "trend": trend.round(2), "seasonal": seasonal.round(2)})
    print(f"Generated {n_months}-month revenue series | mean={revenue.mean():,.0f} | std={revenue.std():,.0f}")
    return df
