# Pearls AQI Predictor

Serverless end-to-end ML pipeline jo aapke city ki Air Quality Index (AQI) agle 3 din
(24h / 48h / 72h) ke liye predict karti hai.

## Architecture

```
[AQICN / OpenWeather API]
        │
        ▼
feature_pipeline/fetch_data.py          -> raw weather + pollutant data
        │
        ▼
feature_pipeline/feature_engineering.py -> time-based + derived features (AQI change rate)
        │
        ▼
feature_pipeline/preprocess.py, feature_Selection.py, validate_features.py
        │
        ▼
feature_pipeline/feature_Store.py       -> Hopsworks Feature Store
        │
        ▼
model_training/train_model.py           -> Random Forest, Ridge, XGBoost, CatBoost,
                                            Neural Network, CatBoost+NN Ensemble
                                            (5 feature sets: FS50/70/90/110/130)
        │
        ▼
register_model.py                       -> Hopsworks Model Registry
        │
        ▼
predict.py                              -> batch inference (24h/48h/72h)
        │
        ▼
app.py (Streamlit)                      -> live dashboard + SHAP + alerts
```

Automation: GitHub Actions (`.github/workflows/`) — feature pipeline hourly,
training pipeline daily.

## Tech Stack

- Python, Scikit-learn, TensorFlow, CatBoost, XGBoost
- Hopsworks (Feature Store + Model Registry)
- GitHub Actions (CI/CD)
- Streamlit (dashboard)
- SHAP (explainability)
- AQICN / OpenWeather API

## Results Summary

Best feature set: **FS70**, Best model: **Ensemble (CatBoost + Neural Network)**

| Horizon | RMSE | MAE  | R²     |
|---------|------|------|--------|
| 24h     | 4.64 | 3.62 | 0.7252 |
| 48h     | 6.72 | 5.18 | 0.4499 |
| 72h     | 8.15 | 6.42 | 0.2274 |
| **Avg** | **6.50** | **5.07** | **0.4675** |

Baseline (persistence) comparison: avg RMSE 9.15, R² -0.063 — model persistence
baseline se significantly behtar hai, especially 48h/72h horizons par.

## How to Run

```bash
# 1. Feature pipeline
python feature_pipeline/fetch_data.py
python feature_pipeline/feature_engineering.py
python feature_pipeline/preprocess.py
python feature_pipeline/validate_features.py

# 2. Training
python model_training/train_model.py
python register_model.py

# 3. Inference
python predict.py

# 4. Dashboard
streamlit run app.py
```

## Project Structure

```
sh_aqi/
├── data/
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
├── register_model.py
├── predict.py
├── app.py
├── requirements.txt
└── .github/workflows/
    ├── feature-pipeline.yml
    └── training-pipeline.yml
```

## Live Dashboard

<!-- Streamlit Community Cloud pe deploy karne ke baad yahan link daalo -->
`https://<your-app>.streamlit.app`