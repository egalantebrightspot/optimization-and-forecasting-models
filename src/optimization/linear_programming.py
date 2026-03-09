"""
Core linear programming primitives, starting with a minimal
resource-allocation model that we can reuse across the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pulp


@dataclass
class ResourceAllocationResult:
    """Solution container for the resource allocation LP.

    Attributes
    ----------
    objective_value:
        Optimal objective value (total cost). ``np.nan`` if no feasible solution.
    allocation:
        2D ``(n_resources, n_products)`` array with allocations ``x[i, j]``.
    status:
        Solver status string from PuLP (e.g. ``"Optimal"``, ``"Infeasible"``).
    resource_slack:
        Remaining capacity for each resource (capacity minus used), shape ``(n_resources,)``.
    product_slack:
        Oversupply for each product (produced minus demand), shape ``(n_products,)``.
    resource_duals:
        Shadow prices for each resource-capacity constraint if available, else ``None``.
    product_duals:
        Shadow prices for each product-demand constraint if available, else ``None``.
    """

    objective_value: float
    allocation: np.ndarray
    status: str
    resource_slack: np.ndarray
    product_slack: np.ndarray
    resource_duals: np.ndarray | None = None
    product_duals: np.ndarray | None = None


def solve_resource_allocation(
    costs: Sequence[float],
    capacities: Sequence[float],
    demands: Sequence[float],
) -> ResourceAllocationResult:
    """Solve a simple resource allocation LP: meet demand at minimum cost.

    Model
    -----
    We assume:
    - ``n_resources = len(capacities)``
    - ``n_products = len(costs) = len(demands)``

    Decision variables:
        ``x[i, j] >= 0`` = quantity of product ``j`` produced using resource ``i``.

    Objective:
        Minimize total cost

        .. math::

            \\min \\sum_i \\sum_j c_j x_{ij}

    Subject to:
        - Resource capacity (each resource has limited total output):

          .. math::

              \\sum_j x_{ij} \\le \\text{capacity}_i \\quad \\forall i

        - Demand satisfaction (each product's demand must be met):

          .. math::

              \\sum_i x_{ij} \\ge \\text{demand}_j \\quad \\forall j

    This is intentionally minimal but already captures classic
    "allocate limited resources to meet product-level demand" logic.
    It also establishes the API pattern for more complex models.
    """

    costs_arr = np.asarray(costs, dtype=float)
    caps_arr = np.asarray(capacities, dtype=float)
    dem_arr = np.asarray(demands, dtype=float)

    if costs_arr.ndim != 1:
        raise ValueError("costs must be a 1D sequence of per-product costs.")
    if caps_arr.ndim != 1:
        raise ValueError("capacities must be a 1D sequence of per-resource capacities.")
    if dem_arr.ndim != 1:
        raise ValueError("demands must be a 1D sequence of per-product demands.")
    if costs_arr.shape[0] != dem_arr.shape[0]:
        raise ValueError(
            f"costs and demands must have the same length; "
            f"got {costs_arr.shape[0]} and {dem_arr.shape[0]}."
        )

    n_resources = caps_arr.shape[0]
    n_products = costs_arr.shape[0]

    # Quick feasibility check: if total capacity < total demand, LP is infeasible.
    if caps_arr.sum() < dem_arr.sum():
        allocation = np.zeros((n_resources, n_products), dtype=float)
        resource_slack = caps_arr.copy()
        product_slack = -dem_arr.copy()
        return ResourceAllocationResult(
            objective_value=np.nan,
            allocation=allocation,
            status="Infeasible (capacity < demand)",
            resource_slack=resource_slack,
            product_slack=product_slack,
            resource_duals=None,
            product_duals=None,
        )

    # Build LP in PuLP.
    prob = pulp.LpProblem("resource_allocation", pulp.LpMinimize)

    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        for i in range(n_resources)
        for j in range(n_products)
    }

    # Objective: sum_i sum_j cost_j * x_ij
    prob += pulp.lpSum(costs_arr[j] * x[(i, j)] for i in range(n_resources) for j in range(n_products))

    # Resource capacity constraints: sum_j x_ij <= capacity_i
    resource_constraints: list[pulp.LpConstraint] = []
    for i in range(n_resources):
        constr = pulp.lpSum(x[(i, j)] for j in range(n_products)) <= caps_arr[i]
        name = f"capacity_{i}"
        prob += constr, name
        resource_constraints.append(prob.constraints[name])

    # Demand satisfaction constraints: sum_i x_ij >= demand_j
    product_constraints: list[pulp.LpConstraint] = []
    for j in range(n_products):
        constr = pulp.lpSum(x[(i, j)] for i in range(n_resources)) >= dem_arr[j]
        name = f"demand_{j}"
        prob += constr, name
        product_constraints.append(prob.constraints[name])

    # Solve with the default CBC solver.
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(prob.status, str(prob.status))

    allocation = np.zeros((n_resources, n_products), dtype=float)
    for (i, j), var in x.items():
        allocation[i, j] = var.value() if var.value() is not None else 0.0

    objective_value = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else np.nan

    # Compute intuitive slack values.
    used_capacity = allocation.sum(axis=1)
    produced = allocation.sum(axis=0)

    resource_slack = caps_arr - used_capacity
    product_slack = produced - dem_arr

    # Duals (shadow prices) – available for CBC via .pi attribute.
    def _extract_duals(constraints: list[pulp.LpConstraint]) -> np.ndarray | None:
        pis = []
        for c in constraints:
            pi = getattr(c, "pi", None)
            if pi is None:
                return None
            pis.append(float(pi))
        return np.asarray(pis, dtype=float)

    resource_duals = _extract_duals(resource_constraints)
    product_duals = _extract_duals(product_constraints)

    return ResourceAllocationResult(
        objective_value=objective_value,
        allocation=allocation,
        status=status,
        resource_slack=resource_slack,
        product_slack=product_slack,
        resource_duals=resource_duals,
        product_duals=product_duals,
    )

