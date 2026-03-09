"""
Baseline time series models and interfaces (starting with ARIMA).
"""

from __future__ import annotations

from typing import Any, Tuple

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

