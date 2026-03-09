from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.forecasting.time_series_models import forecast_horizon, train_arima
from src.optimization.linear_programming import solve_resource_allocation
from src.optimization.resource_allocation import optimize_horizon_from_forecast
from src.api import explain


app = FastAPI(
    title="Optimization and Forecasting API",
    version="0.1.0",
    description="Forecast demand and optimize resource allocation via a simple REST API.",
)

# Mount sub-routers
app.include_router(explain.router)


# ------------------------
# Pydantic models
# ------------------------


class ForecastRequest(BaseModel):
    history: List[float] = Field(..., description="Observed demand history.")
    steps: int = Field(30, description="Number of future periods to forecast.")
    order: Optional[List[int]] = Field(
        default=None,
        description="ARIMA (p,d,q) order. If omitted, defaults to (2,1,2).",
    )


class ForecastResponse(BaseModel):
    forecast: List[float]


class OptimizeRequest(BaseModel):
    costs: List[float] = Field(..., description="Per-unit costs per product.")
    capacities: List[float] = Field(..., description="Maximum capacity per resource.")
    demands: List[float] = Field(..., description="Required demand per product.")


class OptimizeResponse(BaseModel):
    status: str
    objective_value: float
    allocation: List[List[float]]
    resource_slack: List[float]
    product_slack: List[float]


class ForecastAndOptimizeRequest(BaseModel):
    history: List[float] = Field(..., description="Observed demand history.")
    steps: int = Field(30, description="Forecast horizon length.")
    capacities: List[float] = Field(..., description="Maximum capacity per resource.")
    unit_cost: float = Field(1.0, description="Per-unit cost for the (single) product.")
    order: Optional[List[int]] = Field(
        default=None,
        description="ARIMA (p,d,q) order. If omitted, defaults to (2,1,2).",
    )


class HorizonPlanResponse(BaseModel):
    forecast: List[float]
    allocations: List[List[float]]  # shape: (n_periods, n_resources)
    objective_values: List[float]
    statuses: List[str]


class ForecastAndOptimizeResponse(BaseModel):
    forecast: List[float]
    allocations: List[List[float]]
    objective_values: List[float]
    statuses: List[str]


# ------------------------
# Endpoints
# ------------------------


@app.post("/forecast", response_model=ForecastResponse, summary="Forecast future demand")
def forecast_endpoint(payload: ForecastRequest) -> ForecastResponse:
    """Fit an ARIMA model to the provided history and return a forecast."""
    series = pd.Series(payload.history)

    if payload.order is None:
        order = (2, 1, 2)
    else:
        if len(payload.order) != 3:
            raise ValueError("order must have exactly three integers: (p, d, q).")
        order = tuple(int(x) for x in payload.order)

    model = train_arima(series, order=order)
    fc = forecast_horizon(model, steps=payload.steps)
    return ForecastResponse(forecast=[float(x) for x in fc.to_numpy(dtype=float)])


@app.post("/optimize", response_model=OptimizeResponse, summary="Optimize resource allocation")
def optimize_endpoint(payload: OptimizeRequest) -> OptimizeResponse:
    """Solve a basic resource allocation LP for the given costs, capacities, and demands."""
    result = solve_resource_allocation(
        costs=payload.costs,
        capacities=payload.capacities,
        demands=payload.demands,
    )

    return OptimizeResponse(
        status=result.status,
        objective_value=float(result.objective_value),
        allocation=result.allocation.tolist(),
        resource_slack=result.resource_slack.tolist(),
        product_slack=result.product_slack.tolist(),
    )


@app.post(
    "/forecast-and-optimize",
    response_model=ForecastAndOptimizeResponse,
    summary="Forecast demand and optimize capacity over the horizon",
)
def forecast_and_optimize_endpoint(payload: ForecastAndOptimizeRequest) -> ForecastAndOptimizeResponse:
    """End-to-end pipeline: forecast demand, then optimize allocation per period."""
    history_series = pd.Series(payload.history)

    if payload.order is None:
        order = (2, 1, 2)
    else:
        if len(payload.order) != 3:
            raise ValueError("order must have exactly three integers: (p, d, q).")
        order = tuple(int(x) for x in payload.order)

    model = train_arima(history_series, order=order)
    fc = forecast_horizon(model, steps=payload.steps)

    plan = optimize_horizon_from_forecast(
        forecast=fc,
        costs=payload.unit_cost,
        capacities=payload.capacities,
    )

    return ForecastAndOptimizeResponse(
        forecast=[float(x) for x in plan.forecast.to_numpy(dtype=float)],
        allocations=plan.allocations.tolist(),
        objective_values=[float(x) for x in plan.objective_values],
        statuses=plan.statuses,
    )


# The app can be run with:
# uvicorn src.api.main:app --reload

