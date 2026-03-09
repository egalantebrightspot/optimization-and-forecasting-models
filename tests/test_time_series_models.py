import numpy as np
import pandas as pd

from src.data.synthetic_generators import generate_demand_series
from src.forecasting.time_series_models import forecast_horizon, train_arima


def _make_series(n: int, seed: int = 42) -> pd.Series:
    values = generate_demand_series(n_periods=n, random_state=seed)
    index = pd.RangeIndex(start=0, stop=n)
    return pd.Series(values, index=index, name="demand")


def test_train_arima_returns_fitted_model():
    series = _make_series(100)
    model = train_arima(series, order=(1, 1, 1))

    # Basic sanity checks on the fitted model.
    assert hasattr(model, "params")
    assert np.isfinite(model.params).all()


def test_forecast_horizon_returns_series_with_correct_length():
    series = _make_series(120)
    model = train_arima(series, order=(2, 1, 2))

    steps = 24
    fc = forecast_horizon(model, steps=steps)

    assert isinstance(fc, pd.Series)
    assert len(fc) == steps
    assert np.all(np.isfinite(fc.values))

