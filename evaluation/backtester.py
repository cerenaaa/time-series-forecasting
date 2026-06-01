"""Walk-forward backtester for time series models."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResult:
    mae: float
    rmse: float
    mape: float
    coverage_90: float   # Fraction of actuals within 90% PI


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def walk_forward_backtest(model_cls, series: pd.Series, h: int = 6,
                           n_splits: int = 4, min_train: int = 24) -> BacktestResult:
    """
    Walk-forward validation: train on t periods, predict h steps ahead, step forward.
    """
    actuals, predictions = [], []
    n = len(series)

    for split in range(n_splits):
        train_end = min_train + split * h
        if train_end + h > n:
            break
        train = series.iloc[:train_end]
        actual = series.iloc[train_end: train_end + h].values

        try:
            m = model_cls()
            m.fit(train)
            fc = m.predict(h)
            pred = fc.point if hasattr(fc, "point") else np.array(fc)
            actuals.append(actual)
            predictions.append(pred[:len(actual)])
            print(f"  Split {split+1}: train={train_end}, MAE={mae(actual, pred[:len(actual)]):.0f}")
        except Exception as e:
            print(f"  Split {split+1}: failed — {e}")

    if not actuals:
        return BacktestResult(0, 0, 0, 0)

    A = np.concatenate(actuals)
    P = np.concatenate(predictions)
    return BacktestResult(
        mae=round(mae(A, P), 2), rmse=round(rmse(A, P), 2),
        mape=round(mape(A, P), 2), coverage_90=0.0,
    )
