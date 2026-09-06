# Pearls AQI Predictor

A serverless, end-to-end machine learning pipeline that forecasts the Air Quality
Index (AQI) for Karachi over the next three days (24h / 48h / 72h horizons). The
project covers the full lifecycle of a production ML system: automated data
ingestion, feature engineering, a feature store, multi-model training and
selection, a model registry, live inference, and an interactive dashboard with
explainability and alerting — all orchestrated through scheduled CI/CD jobs.

🔗 **Live Dashboard:** https://aqipredictor-syjvh7kjnheplxy2w3pxgm.streamlit.app/
🔗 **GitHub Repository:** https://github.com/shafiaarif/AQI_Predictor
🔗 **Hopsworks Model Registry:** https://eu-west.cloud.hopsworks.ai/p/41176/models

## Table of Contents

- [Overview](#overview)
- [Dashboard Preview](#dashboard-preview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Results Summary](#results-summary)
- [Model Explainability](#model-explainability)
- [Automation (CI/CD)](#automation-cicd)
- [How to Run Locally](#how-to-run-locally)
- [Environment Variables and Secrets](#environment-variables-and-secrets)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)
- [Possible Future Improvements](#possible-future-improvements)

## Overview

Karachi regularly experiences poor air quality, and residents have little
advance warning of how conditions will change over the coming days. This
project builds a fully automated system that:

1. Continuously collects live weather and pollutant data for Karachi.
2. Engineers a rich set of time-series features (lags, rolling statistics,
   trends, pollutant ratios, weather–pollutant interactions, and forward-looking
   weather forecast features).
3. Trains and compares several regression models across multiple feature-set
   sizes to find the best-performing combination.
4. Registers the winning model in a model registry for versioned, reproducible
   deployment.
5. Serves live 24h/48h/72h AQI forecasts through an interactive web dashboard,
   complete with model explainability (SHAP) and hazard alerts.
6. Keeps itself up to date automatically through scheduled GitHub Actions
   workflows, with no manual intervention required after initial setup.

## Dashboard Preview

![Karachi AQI Predictor dashboard — hero card and 3-day forecast](screenshots/dashboard_overview.png)

![SHAP feature importance for the 24h prediction](screenshots/dashboard_shap.png)

The dashboard displays:

- The current AQI for Karachi, color-coded by the standard US AQI category
  (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy,
  Hazardous).
- A 3-day forecast trend chart with shaded AQI category bands for quick visual
  interpretation.
- Individual forecast cards for the 24h, 48h, and 72h horizons, each showing
  the predicted AQI and the direction/magnitude of change from the current
  reading.
- A breakdown of the CatBoost and Neural Network component predictions that
  make up the final ensemble forecast.
- A hazard alert banner that automatically warns when any forecasted horizon
  crosses into an unhealthy AQI range.
- A SHAP-based feature importance chart explaining which inputs are driving
  the 24-hour prediction.

## Architecture

```
Open-Meteo Weather + Air Quality API
        |
        v
feature_pipeline/fetch_data.py
        (fetches raw hourly weather and pollutant data)
        |
        v
feature_pipeline/feature_engineering.py
        (builds lag features, rolling statistics, trend and deviation
        features, pollutant ratios, weather-pollutant interactions, and
        forward-looking weather forecast features)
        |
        v
feature_pipeline/preprocess.py
feature_pipeline/feature_Selection.py
feature_pipeline/validate_features.py
        |
        v
feature_pipeline/feature_Store.py
        (writes the engineered feature set to the Hopsworks Feature Store)
        |
        v
model_training/train_model.py
        (trains Random Forest, Ridge Regression, XGBoost, CatBoost, a Neural
        Network, and a CatBoost + Neural Network ensemble, across five
        feature-set sizes: FS50, FS70, FS90, FS110, FS130)
        |
        v
register_model.py
        (registers the best-performing model artifact in the Hopsworks
        Model Registry)
        |
        v
live_features.py
        (at inference time, merges the full local processed history with a
        live Open-Meteo window to construct a genuine "right now" feature
        row, avoiding the staleness inherent in the training feature view)
        |
        v
predict.py
        (loads the registered model artifacts and produces live 24h/48h/72h
        AQI forecasts)
        |
        v
app.py (Streamlit)
        (renders the live dashboard, including SHAP explainability and
        hazard alerting)
```

### Why a separate live feature builder?

The training feature group drops every row that does not yet have a valid
72-hour-ahead target, since a model cannot be trained on a label it doesn't
have. This means the newest row available in the training feature view is
structurally always about 72 hours behind the current moment. Serving
predictions directly from that feature view would report stale conditions as
"current" on every single request. `live_features.py` solves this by
combining the full historical dataset with a live Open-Meteo fetch and
recomputing the exact same feature formulas used during training, guaranteeing
that live and training features are computed identically and that "now"
genuinely means now.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data source | Open-Meteo Weather & Air Quality APIs |
| Feature store | Hopsworks |
| Modeling | Scikit-learn, XGBoost, CatBoost, TensorFlow/Keras |
| Model registry | Hopsworks |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Automation / CI-CD | GitHub Actions |
| Deployment | Streamlit Community Cloud |

## Results Summary

Five feature-set sizes (50, 70, 90, 110, and 130 engineered features) were
evaluated against seven modeling approaches: a persistence baseline, Random
Forest, Ridge Regression, XGBoost, CatBoost, a Neural Network, and a
CatBoost + Neural Network ensemble.

**Best feature set:** FS70 (70 input features)
**Best model:** Ensemble (CatBoost + Neural Network)

| Horizon | RMSE | MAE  | R²     |
|---------|------|------|--------|
| 24h     | 4.64 | 3.62 | 0.7252 |
| 48h     | 6.72 | 5.18 | 0.4499 |
| 72h     | 8.15 | 6.42 | 0.2274 |
| **Average** | **6.50** | **5.07** | **0.4675** |

For comparison, a naive persistence baseline (predicting no change from the
current reading) achieved an average RMSE of 9.15 and a negative average R²
(-0.063), meaning it performed worse than simply predicting the mean. The
trained ensemble substantially outperforms this baseline at every horizon,
with the largest relative improvement at the 48h and 72h horizons where
persistence degrades the most.

All models predict the **change** in AQI over the forecast horizon rather than
the absolute AQI value. The final forecast is reconstructed as:

```
predicted_aqi = current_aqi + predicted_change
```

## Model Explainability

The dashboard includes a SHAP (SHapley Additive exPlanations) analysis of the
24-hour CatBoost model, showing which input features contribute most to a
given prediction. Recurring top contributors include recent AQI lag values
(reflecting the strong persistence in air quality readings), short-term
changes in PM10 and PM2.5, and the PM2.5-to-AQI ratio, all of which are
consistent with the underlying physical drivers of air quality.

## Automation (CI/CD)

Two scheduled GitHub Actions workflows keep the system up to date without
manual intervention:

- **Feature pipeline** (`.github/workflows/feature-pipeline.yml`) — runs every
  hour, fetching the latest weather and air quality data, engineering
  features, and writing them to the Hopsworks Feature Store.
- **Training pipeline** (`.github/workflows/training-pipeline.yml`) — runs
  weekly, retraining all models across all feature sets and registering the
  best-performing model in the Hopsworks Model Registry. A weekly cadence was
  chosen deliberately, since a full retraining run (five feature sets across
  seven modeling approaches, several with hyperparameter search) takes
  roughly 1.5 hours; both workflows also support manual triggering via
  `workflow_dispatch`.

The dashboard's dependencies are deliberately kept separate from the
pipeline's dependencies (`requirements.txt` vs `requirements-pipeline.txt`),
since the Hopsworks client and Streamlit have conflicting `protobuf` version
requirements that cannot be resolved within a single environment.

## How to Run Locally

```bash
# 1. Feature pipeline
python feature_pipeline/fetch_data.py
python feature_pipeline/feature_engineering.py
python feature_pipeline/preprocess.py
python feature_pipeline/validate_features.py

# 2. Model training
python model_training/train_model.py
python register_model.py

# 3. Live inference (standalone test)
python predict.py

# 4. Dashboard
streamlit run app.py
```

Install dependencies with:

```bash
pip install -r requirements-pipeline.txt   # for the feature and training pipelines
pip install -r requirements.txt            # for the dashboard
```

## Environment Variables and Secrets

The following credentials are required and should be provided as environment
variables (locally) or as repository/app secrets (GitHub Actions and
Streamlit Cloud):

| Variable | Used by | Purpose |
|---|---|---|
| `HOPSWORKS_API_KEY` | feature pipeline, training pipeline, `register_model.py` | Authenticates with the Hopsworks Feature Store and Model Registry |

The dashboard (`app.py`) does not require Hopsworks credentials at runtime,
since `live_features.py` builds inference features from local history plus a
direct Open-Meteo API call rather than querying Hopsworks.

## Project Structure

```
AQI_Predictor/
├── data/
│   └── raw_dataset/
│       └── karachi_processed.csv
├── feature_pipeline/
│   ├── fetch_data.py
│   ├── feature_engineering.py
│   ├── feature_Selection.py
│   ├── feature_Store.py
│   ├── feature_view.py
│   ├── preprocess.py
│   ├── trim_future_rows.py
│   └── validate_features.py
├── model_training/
│   └── train_model.py
├── saved_models/
│   └── fs70/
│       ├── catboost/
│       ├── neural_network/
│       └── feature_columns.json
├── live_features.py
├── predict.py
├── register_model.py
├── app.py
├── requirements.txt
├── requirements-pipeline.txt
├── runtime.txt
├── .streamlit/
│   └── config.toml
└── .github/
    └── workflows/
        ├── feature-pipeline.yml
        └── training-pipeline.yml
```

## Known Limitations

- The local processed history file (`karachi_processed.csv`) is a static
  snapshot. `live_features.py` bridges the gap between this snapshot and the
  current moment by fetching an extended historical window directly from
  Open-Meteo (`past_days`), but the local snapshot should ideally be
  refreshed periodically to keep long-range lag features (up to 21 days)
  fully populated from real historical readings rather than API-fetched
  history alone.
- Forecast accuracy degrades with longer horizons (R² of 0.73 at 24h versus
  0.23 at 72h), which is expected for AQI forecasting given the compounding
  uncertainty of weather forecasts over time.
- The system currently forecasts AQI for a single fixed location (Karachi)
  rather than supporting arbitrary user-specified locations.

## Possible Future Improvements

- Automatically refresh and re-commit the local processed history file as
  part of the hourly feature pipeline, eliminating reliance on a fixed
  `past_days` buffer.
- Extend support to multiple cities with a location selector in the
  dashboard.
- Add automated model performance monitoring and drift detection, retraining
  triggers based on live error metrics rather than a fixed weekly schedule.
- Expose the forecast through a public API endpoint in addition to the
  dashboard, enabling integration with third-party applications.