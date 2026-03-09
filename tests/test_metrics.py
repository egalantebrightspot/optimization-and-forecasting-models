import numpy as np
import pandas as pd

from src.forecasting.time_series_models import forecast_horizon, train_arima
from src.utils.metrics import BacktestResult, mae, mape, rmse, rolling_origin_backtest


def test_point_metrics_basic():
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.0, 1.0, 4.0])

    assert mae(y, y_hat) == 2.0 / 3.0
    assert rmse(y, y_hat) >= mae(y, y_hat)
    assert mape(y, y_hat) > 0.0


def test_rolling_origin_backtest_runs_and_returns_metrics():
    index = pd.RangeIndex(start=0, stop=50)
    # Simple increasing series with small noise.
    y = pd.Series(np.linspace(10.0, 20.0, 50), index=index)

    def _fit(hist: pd.Series):
        return train_arima(hist, order=(1, 1, 0))

    def _forecast(model, steps: int):
        return forecast_horizon(model, steps=steps)

    result = rolling_origin_backtest(
        series=y,
        train_window=20,
        horizon=5,
        fit_fn=_fit,
        forecast_fn=_forecast,
        step=5,
    )

    assert isinstance(result, BacktestResult)
    assert set(result.metrics.index) == {"mae", "rmse", "mape"}
    assert len(result.predictions) > 0

