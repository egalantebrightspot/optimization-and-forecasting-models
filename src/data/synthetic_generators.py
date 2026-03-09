"""
Synthetic data generators for costs, capacities, and demand time series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _get_rng(random_state: Optional[int | np.random.Generator] = None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def generate_costs(
    n_resources: int,
    *,
    base_cost: float = 10.0,
    cost_spread: float = 5.0,
    random_state: Optional[int | np.random.Generator] = None,
) -> np.ndarray:
    """Generate per-unit costs for each resource.

    Costs are positive and mildly heterogeneous across resources.
    """
    if n_resources <= 0:
        raise ValueError("n_resources must be positive.")

    rng = _get_rng(random_state)
    # Draw relative multipliers around 1.0, then scale and clip.
    multipliers = rng.lognormal(mean=0.0, sigma=0.25, size=n_resources)
    costs = base_cost + cost_spread * multipliers
    return costs.astype(float)


def generate_capacities(
    n_resources: int,
    *,
    base_capacity: float = 100.0,
    capacity_spread: float = 50.0,
    random_state: Optional[int | np.random.Generator] = None,
) -> np.ndarray:
    """Generate maximum available capacity for each resource."""
    if n_resources <= 0:
        raise ValueError("n_resources must be positive.")

    rng = _get_rng(random_state)
    raw = rng.lognormal(mean=0.0, sigma=0.5, size=n_resources)
    capacities = base_capacity + capacity_spread * raw
    return capacities.astype(float)


def generate_demand_series(
    n_periods: int,
    *,
    base_level: float = 100.0,
    trend: float = 0.5,
    seasonal_period: int = 7,
    seasonal_amplitude: float = 10.0,
    noise_std: float = 5.0,
    shock_prob: float = 0.05,
    shock_scale: float = 0.3,
    random_state: Optional[int | np.random.Generator] = None,
) -> np.ndarray:
    """Generate a synthetic demand time series with trend, seasonality, noise, and shocks.

    The resulting series is strictly non-negative and shaped ``(n_periods,)``.
    """
    if n_periods <= 0:
        raise ValueError("n_periods must be positive.")
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive.")

    rng = _get_rng(random_state)
    t = np.arange(n_periods, dtype=float)

    # Deterministic components: level + trend + seasonality.
    level_trend = base_level + trend * t
    season = seasonal_amplitude * np.sin(2.0 * np.pi * t / float(seasonal_period))

    # Noise component.
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_periods)

    # Occasional multiplicative shocks (spikes or drops).
    shocks = np.ones(n_periods, dtype=float)
    shock_flags = rng.random(n_periods) < shock_prob
    # Lognormal > 0, centered near 1; subtract 1 so mean effect ~0 then scale.
    shock_factors = rng.lognormal(mean=0.0, sigma=0.75, size=shock_flags.sum())
    shocks[shock_flags] += shock_scale * (shock_factors - 1.0)

    demand = (level_trend + season + noise) * shocks
    # Enforce non-negativity.
    demand = np.clip(demand, a_min=0.0, a_max=None)
    return demand.astype(float)


@dataclass
class ResourceAllocationData:
    """Container for synthetic resource-allocation inputs."""

    costs: np.ndarray
    capacities: np.ndarray
    demand_series: np.ndarray


def generate_resource_allocation_data(
    n_resources: int,
    n_periods: int,
    *,
    random_state: Optional[int | np.random.Generator] = None,
) -> ResourceAllocationData:
    """Generate costs, capacities, and a demand time series for resource-allocation experiments."""
    rng = _get_rng(random_state)

    costs = generate_costs(n_resources, random_state=rng)
    capacities = generate_capacities(n_resources, random_state=rng)
    demand_series = generate_demand_series(n_periods, random_state=rng)

    return ResourceAllocationData(
        costs=costs,
        capacities=capacities,
        demand_series=demand_series,
    )

