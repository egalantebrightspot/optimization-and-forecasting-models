"""
Domain-specific resource allocation models built on top of core optimization primitives.
"""

from __future__ import annotations

from typing import Sequence, Literal

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

