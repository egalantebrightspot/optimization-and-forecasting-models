import pytest
import pandas as pd

from src.optimization.linear_programming import solve_resource_allocation
from src.optimization.resource_allocation import (
    HorizonAllocationPlan,
    optimize_allocation_from_forecast,
    optimize_horizon_from_forecast,
)


def test_solve_resource_allocation_basic_feasible():
    # Simple instance with plenty of capacity.
    costs = [1.0, 3.0]
    capacities = [4.0, 2.0]  # total capacity 6
    demands = [2.0, 1.0]  # total demand 3

    result = solve_resource_allocation(costs=costs, capacities=capacities, demands=demands)

    # Status and objective
    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(5.0, rel=1e-6)

    # Demand must be satisfied (produced >= demand per product)
    produced_per_product = result.allocation.sum(axis=0)
    assert produced_per_product[0] >= demands[0] - 1e-6
    assert produced_per_product[1] >= demands[1] - 1e-6

    # Capacity should not be exceeded
    used_capacity = result.allocation.sum(axis=1)
    for used, cap in zip(used_capacity, capacities):
        assert used <= cap + 1e-6


def test_optimize_allocation_from_forecast_mean_aggregation():
    # Simple increasing forecast over 5 periods.
    forecast = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    capacities = [20.0, 20.0]

    result = optimize_allocation_from_forecast(
        forecast=forecast,
        capacities=capacities,
        unit_cost=2.0,
        aggregation="mean",
    )

    assert result.status == "Optimal"
    # Mean forecast is 14.0 units; total allocation should be close to that.
    total_alloc = result.allocation.sum()
    assert total_alloc == pytest.approx(14.0, rel=1e-6)


def test_optimize_horizon_from_forecast_per_period():
    forecast = pd.Series([5.0, 7.0, 9.0])
    capacities = [10.0, 10.0]

    plan = optimize_horizon_from_forecast(
        forecast=forecast,
        costs=1.5,
        capacities=capacities,
    )

    assert isinstance(plan, HorizonAllocationPlan)
    assert plan.allocations.shape == (len(forecast), len(capacities))

    # For each period, total allocated quantity should match demand.
    per_period_alloc = plan.allocations.sum(axis=1)
    assert per_period_alloc.tolist() == pytest.approx(forecast.tolist(), rel=1e-6)


