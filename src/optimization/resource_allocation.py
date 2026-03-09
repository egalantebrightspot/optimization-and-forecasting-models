"""
Domain-specific resource allocation models built on top of core optimization primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Literal, Union, List
import numpy as np
import pandas as pd

from .linear_programming import ResourceAllocationResult, solve_resource_allocation


def optimize_allocation_from_forecast(
    forecast: pd.Series,
    capacities: Sequence[float],
    *,
    unit_cost: float = 1.0,
    aggregation: Literal["mean", "sum", "max"] = "mean",
) -> ResourceAllocationResult:
    """Optimize single-product resource allocation given a demand forecast.

    Parameters
    ----------
    forecast:
        Forecasted demand over a planning horizon as a ``pandas.Series``.
    capacities:
        Maximum capacity for each resource; will be broadcast to a single product.
    unit_cost:
        Per-unit cost for producing the product (same across resources in this
        minimal model).
    aggregation:
        How to turn the forecast horizon into a scalar planning requirement:
        ``\"mean\"`` (default), ``\"sum\"``, or ``\"max\"``.

    Returns
    -------
    ResourceAllocationResult
        Solution of the underlying linear program.
    """
    if not isinstance(forecast, pd.Series):
        raise TypeError("forecast must be a pandas.Series.")
    if forecast.empty:
        raise ValueError("forecast must be non-empty.")

    forecast_values = forecast.to_numpy(dtype=float)

    if aggregation == "mean":
        required_demand = float(np.mean(forecast_values))
    elif aggregation == "sum":
        required_demand = float(np.sum(forecast_values))
    elif aggregation == "max":
        required_demand = float(np.max(forecast_values))
    else:
        raise ValueError('aggregation must be one of {"mean", "sum", "max"}.')

    costs = [float(unit_cost)]
    demands = [required_demand]

    return solve_resource_allocation(
        costs=costs,
        capacities=list(capacities),
        demands=demands,
    )


@dataclass
class HorizonAllocationPlan:
    """Allocation plan over a forecast horizon."""

    forecast: pd.Series
    allocations: np.ndarray  # shape: (n_periods, n_resources)
    objective_values: np.ndarray  # shape: (n_periods,)
    statuses: List[str]


def optimize_horizon_from_forecast(
    forecast: pd.Series,
    costs: Union[float, Sequence[float]],
    capacities: Sequence[float],
) -> HorizonAllocationPlan:
    """Optimize resource allocation for each period in a forecast horizon.

    This treats the forecast as demand for a **single product** over time and
    solves one LP per period using ``solve_resource_allocation``.
    """
    if not isinstance(forecast, pd.Series):
        raise TypeError("forecast must be a pandas.Series.")
    if forecast.empty:
        raise ValueError("forecast must be non-empty.")

    if isinstance(costs, (int, float)):
        cost_list = [float(costs)]
    else:
        cost_list = list(costs)

    if len(cost_list) != 1:
        raise ValueError("optimize_horizon_from_forecast currently supports a single product (len(costs) must be 1).")

    capacities_list = list(capacities)
    n_resources = len(capacities_list)
    n_periods = len(forecast)

    allocations = np.zeros((n_periods, n_resources), dtype=float)
    objective_values = np.zeros(n_periods, dtype=float)
    statuses: List[str] = []

    for t, demand_t in enumerate(forecast.to_numpy(dtype=float)):
        period_demand = max(float(demand_t), 0.0)
        res = solve_resource_allocation(
            costs=cost_list,
            capacities=capacities_list,
            demands=[period_demand],
        )
        allocations[t, :] = res.allocation[:, 0]
        objective_values[t] = res.objective_value
        statuses.append(res.status)

    return HorizonAllocationPlan(
        forecast=forecast.copy(),
        allocations=allocations,
        objective_values=objective_values,
        statuses=statuses,
    )


