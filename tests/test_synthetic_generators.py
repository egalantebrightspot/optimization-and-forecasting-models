import numpy as np

from src.data.synthetic_generators import (
    ResourceAllocationData,
    generate_capacities,
    generate_costs,
    generate_demand_series,
    generate_resource_allocation_data,
)


def test_generate_demand_series_shape_and_non_negative():
    n = 365
    series = generate_demand_series(n, random_state=42)
    assert isinstance(series, np.ndarray)
    assert series.shape == (n,)
    assert np.all(series >= 0.0)


def test_generate_demand_series_reproducible():
    s1 = generate_demand_series(100, random_state=123)
    s2 = generate_demand_series(100, random_state=123)
    assert np.allclose(s1, s2)


def test_generate_costs_and_capacities_shapes_and_positive():
    n_resources = 5
    rng_seed = 7
    costs = generate_costs(n_resources, random_state=rng_seed)
    caps = generate_capacities(n_resources, random_state=rng_seed)

    assert costs.shape == (n_resources,)
    assert caps.shape == (n_resources,)
    assert np.all(costs > 0.0)
    assert np.all(caps > 0.0)


def test_generate_resource_allocation_data_consistency():
    data = generate_resource_allocation_data(n_resources=3, n_periods=50, random_state=99)

    assert isinstance(data, ResourceAllocationData)
    assert data.costs.shape == (3,)
    assert data.capacities.shape == (3,)
    assert data.demand_series.shape == (50,)

