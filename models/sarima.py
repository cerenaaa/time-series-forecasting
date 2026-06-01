"""SARIMA forecaster with automatic order selection via AIC."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import product

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@dataclass
class SARIMAForecast:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    order: tuple
    seasonal_order: tuple
    aic: float


class AutoSARIMA:
    def __init__(self, seasonal_period: int = 12, max_p: int = 2, max_q: int = 2):
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_q = max_q
        self.model = None
        self.result = None
        self.best_order = None
        self.best_seasonal_order = None

    def _select_order(self, series: np.ndarray) -> tuple:
        best_aic, best_order, best_seasonal = np.inf, (1, 1, 1), (1, 1, 0, 12)
        for p, q in product(range(self.max_p + 1), range(self.max_q + 1)):
            try:
                m = SARIMAX(series, order=(p, 1, q),
                            seasonal_order=(1, 1, 0, self.seasonal_period),
                            enforce_stationarity=False).fit(disp=False)
                if m.aic < best_aic:
                    best_aic, best_order = m.aic, (p, 1, q)
            except Exception:
                continue
        return best_order, best_seasonal

    def fit(self, series: pd.Series) -> "AutoSARIMA":
        if not STATSMODELS_AVAILABLE:
            raise ImportError("pip install statsmodels")
        self.best_order, self.best_seasonal_order = self._select_order(series.values)
        self.model = SARIMAX(series, order=self.best_order,
                             seasonal_order=self.best_seasonal_order,
                             enforce_stationarity=False)
        self.result = self.model.fit(disp=False)
        print(f"SARIMA{self.best_order}x{self.best_seasonal_order} | AIC={self.result.aic:.1f}")
        return self

    def predict(self, h: int = 12, alpha: float = 0.05) -> SARIMAForecast:
        forecast = self.result.get_forecast(steps=h)
        ci = forecast.conf_int(alpha=alpha)
        return SARIMAForecast(
            point=forecast.predicted_mean.values,
            lower=ci.iloc[:, 0].values,
            upper=ci.iloc[:, 1].values,
            order=self.best_order,
            seasonal_order=self.best_seasonal_order,
            aic=self.result.aic,
        )
