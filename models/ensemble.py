"""
Forecast ensemble: combines multiple models using inverse-error weighting.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class EnsembleForecast:
    point: np.ndarray
    weights: dict[str, float]


class ForecastEnsemble:
    def __init__(self):
        self.forecasters: dict = {}
        self.val_errors: dict = {}

    def add(self, name: str, forecaster, val_error: float):
        self.forecasters[name] = forecaster
        self.val_errors[name] = val_error
        return self

    def predict(self, h: int = 12) -> EnsembleForecast:
        inv_errors = {k: 1 / (v + 1e-9) for k, v in self.val_errors.items()}
        total = sum(inv_errors.values())
        weights = {k: v / total for k, v in inv_errors.items()}

        combined = np.zeros(h)
        for name, forecaster in self.forecasters.items():
            forecast = forecaster.predict(h)
            point = forecast.point if hasattr(forecast, "point") else forecast
            combined += weights[name] * point[:h]

        print("Ensemble weights:", {k: round(v, 3) for k, v in weights.items()})
        return EnsembleForecast(point=combined, weights=weights)
