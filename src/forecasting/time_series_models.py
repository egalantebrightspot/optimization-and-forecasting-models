"""
Baseline time series models and interfaces (starting with ARIMA).
"""

from __future__ import annotations

from typing import Any, Tuple, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper


def train_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (2, 1, 2),
    **fit_kwargs: Any,
) -> ARIMAResultsWrapper:
    """Fit an ARIMA model to a univariate time series.

    Parameters
    ----------
    series:
        Univariate ``pandas.Series`` of observations. The index (e.g. dates)
        is preserved on the fitted model for use in forecasting.
    order:
        ARIMA ``(p, d, q)`` order; defaults to ``(2, 1, 2)`` as a reasonable
        generic baseline.
    **fit_kwargs:
        Additional keyword arguments forwarded to ``model.fit()``.

    Returns
    -------
    ARIMAResultsWrapper
        Fitted statsmodels ARIMA results object.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas.Series.")
    if series.empty:
        raise ValueError("series must be non-empty.")

    model = ARIMA(series, order=order)
    fitted: ARIMAResultsWrapper = model.fit(**fit_kwargs)
    return fitted


def forecast_horizon(
    model: ARIMAResultsWrapper,
    steps: int = 30,
) -> pd.Series:
    """Forecast ``steps`` periods into the future from a fitted ARIMA model."""
    if steps <= 0:
        raise ValueError("steps must be positive.")

    forecast_res = model.get_forecast(steps=steps)
    mean_forecast: pd.Series = forecast_res.predicted_mean
    # Ensure a plain Series is returned, not an array-like.
    if not isinstance(mean_forecast, pd.Series):
        mean_forecast = pd.Series(mean_forecast)
    return mean_forecast


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """Compute simple accuracy metrics between actual and predicted series.

    Both series are aligned on their index; any non-overlapping points or
    NaNs are dropped before metric computation.

    Returns a ``pandas.Series`` with metrics such as MAE, RMSE, and MAPE.
    """
    if not isinstance(actual, pd.Series) or not isinstance(predicted, pd.Series):
        raise TypeError("actual and predicted must both be pandas.Series.")

    df = pd.concat({"actual": actual, "predicted": predicted}, axis=1).dropna()
    if df.empty:
        raise ValueError("No overlapping, non-NaN data points to evaluate.")

    y = df["actual"].to_numpy(dtype=float)
    y_hat = df["predicted"].to_numpy(dtype=float)

    errors = y_hat - y
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # Avoid division by zero in MAPE by flooring the denominator.
    denom = np.maximum(np.abs(y), 1e-8)
    mape = float(np.mean(np.abs(errors) / denom) * 100.0)

    return pd.Series(
        {"mae": mae, "rmse": rmse, "mape": mape},
        name="forecast_metrics",
    )

