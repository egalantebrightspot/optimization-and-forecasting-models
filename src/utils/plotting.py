"""
Plotting utilities for optimization and forecasting results.
"""

from __future__ import annotations

from typing import Sequence, Optional

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

