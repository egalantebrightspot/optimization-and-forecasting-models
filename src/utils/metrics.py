"""
Common metrics used across optimization and forecasting workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape.")
    return float(np.mean(np.abs(predicted - actual)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape.")
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, *, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (in percent)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape.")
    denom = np.maximum(np.abs(actual), eps)
    return float(np.mean(np.abs(predicted - actual) / denom) * 100.0)


@dataclass
class BacktestResult:
    metrics: pd.Series
    predictions: pd.Series


def rolling_origin_backtest(
    series: pd.Series,
    train_window: int,
    horizon: int,
    fit_fn,
    forecast_fn,
    *,
    step: int = 1,
) -> BacktestResult:
    """Simple rolling-origin backtest for univariate series.

    Parameters
    ----------
    series:
        Univariate ``pandas.Series`` to backtest on.
    train_window:
        Number of observations in each training window.
    horizon:
        Forecast horizon (number of steps ahead) at each origin.
    fit_fn:
        Callable ``fit_fn(history: pd.Series) -> model``.
    forecast_fn:
        Callable ``forecast_fn(model, steps: int) -> pd.Series``.
    step:
        Step size between successive forecast origins.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas.Series.")
    if train_window <= 0 or horizon <= 0 or step <= 0:
        raise ValueError("train_window, horizon, and step must be positive.")
    if len(series) <= train_window + horizon:
        raise ValueError("series too short for given train_window and horizon.")

    preds = []
    actuals = []

    last_start = len(series) - train_window - horizon
    for start in range(0, last_start + 1, step):
        train_end = start + train_window
        hist = series.iloc[start:train_end]
        model = fit_fn(hist)
        fc = forecast_fn(model, steps=horizon)

        # Align forecast with true future values.
        fc_index = series.index[train_end : train_end + horizon]
        fc = pd.Series(fc.values, index=fc_index)
        y_true = series.loc[fc_index]

        preds.append(fc)
        actuals.append(y_true)

    y_pred = pd.concat(preds).sort_index()
    y_true = pd.concat(actuals).sort_index()

    metrics = pd.Series(
        {
            "mae": mae(y_true.values, y_pred.values),
            "rmse": rmse(y_true.values, y_pred.values),
            "mape": mape(y_true.values, y_pred.values),
        },
        name="backtest_metrics",
    )
    return BacktestResult(metrics=metrics, predictions=y_pred)

