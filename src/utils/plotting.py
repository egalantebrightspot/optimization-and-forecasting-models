"""
Plotting utilities for optimization and forecasting results.
"""

from __future__ import annotations

from typing import Sequence, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimization.linear_programming import ResourceAllocationResult


def plot_demand_series(
    series: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Demand over time",
    ylabel: str = "Units",
) -> plt.Axes:
    """Plot a single demand time series."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


def plot_forecast_vs_actual(
    history: pd.Series,
    actual_future: pd.Series,
    forecast: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Forecast vs actual",
    ylabel: str = "Units",
) -> plt.Axes:
    """Plot historical data, actual future values, and forecast on one chart."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    history.plot(ax=ax, label="history")
    actual_future.plot(ax=ax, label="actual future")
    forecast.plot(ax=ax, label="forecast", linestyle="--")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def plot_allocation_bars(
    result: ResourceAllocationResult,
    *,
    resource_labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Resource capacity vs allocated production",
    ylabel: str = "Units",
) -> plt.Axes:
    """Bar chart comparing capacity and allocation per resource."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n_resources = result.allocation.shape[0]
    if resource_labels is None:
        resource_labels = [f"R{i+1}" for i in range(n_resources)]

    allocated = result.allocation.sum(axis=1)
    capacity_values = allocated + result.resource_slack

    resources = np.arange(n_resources)
    width = 0.35

    ax.bar(resources - width / 2, capacity_values, width, label="capacity")
    ax.bar(resources + width / 2, allocated, width, label="allocated")
    ax.set_xticks(resources)
    ax.set_xticklabels(list(resource_labels))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_allocation_heatmap(
    allocations: np.ndarray,
    *,
    index: Optional[pd.Index] = None,
    resource_labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Allocation heatmap (period x resource)",
    xlabel: str = "Period",
    ylabel: str = "Resource",
) -> plt.Axes:
    """Heatmap of allocation levels for each resource over time.

    ``allocations`` is expected to have shape ``(n_periods, n_resources)``.
    """
    if allocations.ndim != 2:
        raise ValueError("allocations must be a 2D array of shape (n_periods, n_resources).")

    n_periods, n_resources = allocations.shape
    if resource_labels is None:
        resource_labels = [f"R{i+1}" for i in range(n_resources)]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(
        allocations.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(n_resources))
    ax.set_yticklabels(list(resource_labels))

    if index is not None:
        ax.set_xticks(np.arange(n_periods))
        ax.set_xticklabels([str(i) for i in index], rotation=45, ha="right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Allocation")
    return ax


def plot_objective_over_time(
    objective: Union[pd.Series, np.ndarray],
    *,
    index: Optional[pd.Index] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Objective value over time",
    ylabel: str = "Cost",
) -> plt.Axes:
    """Line plot of objective value (e.g., cost) over time."""
    if isinstance(objective, pd.Series):
        series = objective
    else:
        values = np.asarray(objective, dtype=float)
        if index is None:
            index = pd.RangeIndex(start=0, stop=len(values))
        series = pd.Series(values, index=index, name="objective")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    series.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


