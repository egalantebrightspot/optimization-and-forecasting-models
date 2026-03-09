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
  api/
    main.py                    # FastAPI app (forecast, optimize, explain)
    explain.py                 # Forecast+optimize explanation endpoint
notebooks/
  01_linear_programming_resource_allocation.ipynb
  02_nonlinear_optimization_capacity_planning.ipynb
  03_time_series_demand_forecasting.ipynb
```

---

## Quickstart

### Environment and installation

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
python -m pip install -r requirements.txt
pytest                    # run test suite
```

### Core workflow: forecast → optimize

The core decision loop is:

- **Generate synthetic demand** (`src/data/synthetic_generators.py`)
- **Fit ARIMA and forecast** (`src/forecasting/time_series_models.py`)
- **Optimize resource allocation** (`src/optimization/resource_allocation.py`)

Minimal example:

```python
import pandas as pd

from src.data.synthetic_generators import generate_demand_series
from src.forecasting.time_series_models import train_arima, forecast_horizon
from src.optimization.resource_allocation import optimize_horizon_from_forecast

values = generate_demand_series(200, random_state=0)
series = pd.Series(values)

model = train_arima(series, order=(2, 1, 2))
forecast = forecast_horizon(model, steps=30)

plan = optimize_horizon_from_forecast(
    forecast=forecast,
    costs=1.0,
    capacities=[150.0, 120.0, 90.0],
)
```

---

## Notebooks

- `03_time_series_demand_forecasting.ipynb` is the **flagship demo**, showing:
  - synthetic demand generation
  - ARIMA forecasting
  - forecast → optimization with `optimize_horizon_from_forecast`
  - allocation and cost visualization over the horizon

Launch notebooks with:

```bash
jupyter lab
```

---

## REST API (FastAPI + Swagger)

An HTTP API exposes the same workflow for external clients.

Start the server from the project root:

```bash
uvicorn src.api.main:app --reload --port 8001
```

Then open:

- Swagger UI: `http://127.0.0.1:8001/docs`
- OpenAPI JSON: `http://127.0.0.1:8001/openapi.json`

Key endpoints:

- `POST /forecast` – fit ARIMA to a history and return a forecast.
- `POST /optimize` – solve a resource allocation LP for given costs, capacities, and demands.
- `POST /forecast-and-optimize` – end-to-end forecast → horizon optimization.
- `POST /explain/` – run forecast and optimization and return a structured explanation of the plan.