import pytest

from src.optimization.linear_programming import solve_resource_allocation


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

