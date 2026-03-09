# Optimization and Forecasting Models

Mathematics-driven project focused on **optimization** and **forecasting** for quantitative decision systems. This repository demonstrates how to translate linear and nonlinear programming, resource allocation, and time series forecasting into **reproducible Python architectures** that support real-world planning and decision intelligence.

The goal is to show how optimization and forecasting can be combined into a coherent decision system:  
- optimization chooses the best actions under constraints  
- forecasting provides the demand, load, or risk inputs those decisions depend on  

---

## Objectives

- Implement **linear programming** and **nonlinear optimization** models for resource allocation and scheduling.
- Build **time series forecasting** pipelines for demand and load prediction.
- Use **synthetic but realistic data** to simulate operational scenarios.
- Provide **notebooks** that walk through end-to-end decision workflows.
- Structure the code as a **reusable library** suitable for production integration.

---

## Project structure

```text
src/
  optimization/
    linear_programming.py      # LP models (e.g., resource allocation, capacity planning)
    nonlinear_optimization.py  # Nonlinear and constrained optimization examples
    resource_allocation.py     # Domain-specific formulations
    scheduling_models.py       # Simple scheduling / assignment models
  forecasting/
    time_series_models.py      # ARIMA/ETS/Prophet-style interfaces
    demand_forecasting.py      # Demand/load forecasting workflows
    evaluation.py              # MAPE, RMSE, backtesting utilities
  data/
    synthetic_generators.py    # Synthetic demand, capacity, and cost data
    loaders.py                 # CSV/parquet loaders and schema helpers
  utils/
    plotting.py                # Plots for forecasts, residuals, capacity vs demand
    metrics.py                 # Shared metrics for optimization and forecasting
notebooks/
  01_linear_programming_resource_allocation.ipynb
  02_nonlinear_optimization_capacity_planning.ipynb
  03_time_series_demand_forecasting.ipynb