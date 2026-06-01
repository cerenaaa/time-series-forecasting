"""
STL decomposition: separates trend, seasonality, and residual components.
Useful for understanding series structure and as a forecasting baseline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DecompositionResult:
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    seasonal_strength: float   # Var(seasonal) / Var(Y - trend)
    trend_strength: float


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(x, np.nan, dtype=float)
    half = window // 2
    for i in range(half, len(x) - half):
        result[i] = np.mean(x[i - half: i + half + 1])
    return result


def extract_seasonality(detrended: np.ndarray, period: int) -> np.ndarray:
    seasonal = np.zeros_like(detrended)
    for s in range(period):
        indices = range(s, len(detrended), period)
        valid = [detrended[i] for i in indices if not np.isnan(detrended[i])]
        if valid:
            avg = np.mean(valid)
            for i in indices:
                seasonal[i] = avg
    return seasonal - seasonal.mean()


def decompose(series: np.ndarray, period: int = 12) -> DecompositionResult:
    trend = moving_average(series, period)
    valid = ~np.isnan(trend)
    detrended = np.where(valid, series - trend, np.nan)
    seasonal = extract_seasonality(detrended, period)
    residual = series - np.nan_to_num(trend, nan=np.nanmean(trend)) - seasonal

    var_y_minus_trend = np.var(series[valid] - trend[valid])
    seasonal_strength = max(0, 1 - np.var(residual[valid]) / var_y_minus_trend) if var_y_minus_trend > 0 else 0
    trend_strength = max(0, 1 - np.var(residual[valid]) / np.var((series - seasonal)[valid])) if valid.sum() > 0 else 0

    return DecompositionResult(
        trend=trend, seasonal=seasonal, residual=residual,
        seasonal_strength=round(seasonal_strength, 3),
        trend_strength=round(trend_strength, 3),
    )
