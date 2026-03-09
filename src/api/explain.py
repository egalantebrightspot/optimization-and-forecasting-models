from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.forecasting.time_series_models import forecast_horizon, train_arima
from src.optimization.resource_allocation import optimize_horizon_from_forecast, HorizonAllocationPlan


router = APIRouter(prefix="/explain", tags=["Explain"])


class ExplainRequest(BaseModel):
    history: List[float] = Field(..., description="Observed demand history.")
    steps: int = Field(..., description="Forecast horizon length.")
    capacities: List[float] = Field(..., description="Maximum capacity per resource.")
    unit_cost: float = Field(1.0, description="Per-unit cost for the (single) product.")
    order: Tuple[int, int, int] = Field(
        (2, 1, 2),
        description="ARIMA (p,d,q) order.",
    )


@router.post("/", summary="Explain forecast and optimization plan")
def explain_plan(req: ExplainRequest):
    series = pd.Series(req.history)

    # Forecast
    model = train_arima(series, order=req.order)
    forecast = forecast_horizon(model, req.steps)

    # Optimization
    plan = optimize_horizon_from_forecast(
        forecast=forecast,
        costs=req.unit_cost,
        capacities=req.capacities,
    )

    # Explanation
    explanation = build_explanation(plan, capacities=req.capacities)

    return {
        "forecast": forecast.tolist(),
        "allocations": plan.allocations.tolist(),
        "objective_values": plan.objective_values.tolist(),
        "statuses": plan.statuses,
        "explanation": explanation,
    }


def build_explanation(plan: HorizonAllocationPlan, capacities: List[float]):
    allocations = plan.allocations
    n_periods, n_resources = allocations.shape

    caps = np.asarray(capacities, dtype=float)
    caps = np.where(caps <= 0.0, 1.0, caps)  # avoid divide-by-zero

    # Resource utilization
    utilization = allocations / caps
    avg_util = utilization.mean(axis=0)
    max_util = utilization.max(axis=0)

    # Bottleneck resource
    bottleneck = int(np.argmax(max_util))

    # Cost behavior
    cost_trend = "increasing" if plan.objective_values[-1] > plan.objective_values[0] else "stable"

    # Demand fulfillment based on solver statuses (tolerate case variants)
    demand_fulfillment = (
        "all periods satisfied"
        if all(status.lower() == "optimal" for status in plan.statuses)
        else "some periods infeasible"
    )

    explanation = {
        "resource_summary": [
            {
                "resource": i,
                "average_utilization": float(avg_util[i]),
                "peak_utilization": float(max_util[i]),
                "at_capacity": bool(max_util[i] >= 0.99),
            }
            for i in range(n_resources)
        ],
        "bottleneck_resource": bottleneck,
        "demand_fulfillment": demand_fulfillment,
        "cost_behavior": cost_trend,
        "overall_interpretation": (
            f"Resource {bottleneck} is the primary bottleneck. "
            f"Costs appear {cost_trend} over the forecast horizon. "
            "Utilization patterns suggest where capacity expansions would have the greatest impact."
        ),
    }

    return explanation

