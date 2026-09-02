"""
predict.py
----------

Live Karachi AQI features (via live_features.py) se FS70 ensemble
models se 24h / 48h / 72h AQI predict karta hai.

CHANGED: pehle yeh Hopsworks Feature View se "latest row" fetch karta
tha — jo hamesha ~72h purana hota hai (training feature group targets
ke liye dropna() karta hai). Ab yeh live_features.build_live_feature_row()
use karta hai, jo local history + Open-Meteo ka live forecast merge
karke asal "abhi" wala row banata hai.

Actual saved model structure:

saved_models/
└── fs70/
    ├── catboost/
    │   ├── target_aqi_24.pkl
    │   ├── target_aqi_48.pkl
    │   └── target_aqi_72.pkl
    │
    ├── neural_network/
    │   ├── model.keras
    │   └── scaler.pkl
    │
    └── feature_columns.json
"""

import os
import json
import pickle
import pandas as pd
import joblib

from catboost import CatBoostRegressor
from tensorflow import keras

from live_features import build_live_feature_row


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models", "fs70")
CATBOOST_DIR = os.path.join(MODEL_DIR, "catboost")
NN_DIR = os.path.join(MODEL_DIR, "neural_network")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

HORIZONS = ["24h", "48h", "72h"]

TARGET_COLUMNS = [
    "target_aqi_24",
    "target_aqi_48",
    "target_aqi_72",
]

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    "catboost": 0.5,
    "nn": 0.5,
}


# ============================================================
# CHECK FILES
# ============================================================

def check_required_files():
    """
    Make sure all required model files exist.
    """

    required_files = [
        os.path.join(CATBOOST_DIR, "target_aqi_24.pkl"),
        os.path.join(CATBOOST_DIR, "target_aqi_48.pkl"),
        os.path.join(CATBOOST_DIR, "target_aqi_72.pkl"),
        os.path.join(NN_DIR, "model.keras"),
        os.path.join(NN_DIR, "scaler.pkl"),
        FEATURE_COLUMNS_PATH,
    ]

    print("=" * 60)
    print("CHECKING MODEL FILES")
    print("=" * 60)

    for path in required_files:
        if os.path.exists(path):
            print(f"[OK] {path}")
        else:
            raise FileNotFoundError(f"\nRequired file not found:\n{path}")

    print("All required model files found!")


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

def load_feature_columns():
    """
    Load feature order used during model training.
    """

    print("\n" + "=" * 60)
    print("LOADING FEATURE COLUMNS")
    print("=" * 60)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        feature_columns = data
    elif isinstance(data, dict):
        if "feature_columns" in data:
            feature_columns = data["feature_columns"]
        elif "features" in data:
            feature_columns = data["features"]
        else:
            raise ValueError(
                "feature_columns.json does not contain "
                "'feature_columns' or 'features'."
            )
    else:
        raise ValueError("Unexpected format in feature_columns.json")

    print(f"Number of training features: {len(feature_columns)}")

    return feature_columns


# ============================================================
# GET LIVE FEATURES (replaces old Hopsworks Feature View fetch)
# ============================================================

def get_latest_features(feature_columns):
    """
    Build the current "right now" feature row using live_features.py
    (local history + live Open-Meteo forecast), instead of reading a
    stale row from the Hopsworks Feature View.

    Returns:
        row_df      : single-row DataFrame, already restricted + ordered
                      to match feature_columns.
        current_aqi : float, actual us_aqi as of "now".
    """

    print("\n" + "=" * 60)
    print("BUILDING LIVE FEATURES")
    print("=" * 60)

    row_df, current_aqi, now_timestamp = build_live_feature_row(
        feature_columns_path=FEATURE_COLUMNS_PATH,
        verbose=True,
    )

    print(f"\n'Now' timestamp : {now_timestamp}")
    print(f"Current AQI     : {current_aqi:.1f}")

    return row_df, current_aqi


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():
    """
    Load:

    CatBoost:
        target_aqi_24.pkl
        target_aqi_48.pkl
        target_aqi_72.pkl

    Neural Network:
        model.keras
        scaler.pkl
    """

    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    models = {
        "catboost": {},
        "nn": None,
        "scaler": None,
    }

    print("\nLoading CatBoost models...")

    for horizon in HORIZONS:
        target_name = f"target_aqi_{horizon.replace('h', '')}"
        cb_path = os.path.join(CATBOOST_DIR, f"{target_name}.pkl")

        print(f"Loading {cb_path}")

        try:
            with open(cb_path, "rb") as f:
                cb_model = pickle.load(f)
        except Exception:
            print("Pickle loading failed, trying CatBoost native loader...")
            cb_model = CatBoostRegressor()
            cb_model.load_model(cb_path)

        models["catboost"][horizon] = cb_model
        print(f"[OK] CatBoost {horizon}")

    nn_path = os.path.join(NN_DIR, "model.keras")
    scaler_path = os.path.join(NN_DIR, "scaler.pkl")

    print("\nLoading Neural Network...")
    print(nn_path)
    nn_model = keras.models.load_model(nn_path)
    print("[OK] Neural Network loaded")

    print("\nLoading Neural Network scaler...")
    print(scaler_path)
    scaler = joblib.load(scaler_path)
    print("[OK] Scaler loaded")

    models["nn"] = nn_model
    models["scaler"] = scaler

    return models


# ============================================================
# PREPARE INPUT FEATURES
# ============================================================

def prepare_features(row_df, feature_columns):
    """
    live_features.build_live_feature_row() already returns a row
    restricted + ordered to feature_columns with no NaNs, so this is
    now just a final numeric-safety check before feeding the models.
    """

    print("\n" + "=" * 60)
    print("PREPARING INPUT FEATURES")
    print("=" * 60)

    X = row_df[feature_columns].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    print(f"Final input shape: {X.shape}")
    print(f"Expected feature count: {len(feature_columns)}")

    if X.isna().any().any():
        nan_columns = X.columns[X.isna().any()].tolist()
        raise ValueError("NaN values found in model input: " + str(nan_columns))

    return X


# ============================================================
# PREDICT
# ============================================================

def predict_aqi(row_df, current_aqi, models, feature_columns):
    """
    Generate 24h / 48h / 72h predictions.

    Models predict AQI CHANGE (not absolute).
    Final AQI = current_aqi + predicted_change.
    """

    print("\n" + "=" * 60)
    print("GENERATING AQI FORECAST")
    print("=" * 60)

    print(f"Current AQI (us_aqi): {current_aqi:.1f}")

    X = prepare_features(row_df, feature_columns)

    # --------------------------------------------------------
    # CatBoost predictions (these are CHANGE values)
    # --------------------------------------------------------
    catboost_predictions = {}
    print("\nCatBoost predictions (change):")
    for horizon in HORIZONS:
        model = models["catboost"][horizon]
        prediction = float(model.predict(X)[0])
        catboost_predictions[horizon] = prediction
        print(f"  {horizon}: {prediction:.2f}")

    # --------------------------------------------------------
    # Neural Network prediction (also CHANGE values)
    # --------------------------------------------------------
    print("\nNeural Network prediction...")
    scaler = models["scaler"]
    X_scaled = scaler.transform(X)
    nn_raw = models["nn"].predict(X_scaled, verbose=0)

    nn_prediction = nn_raw[0]
    if hasattr(nn_prediction, "tolist"):
        nn_prediction = nn_prediction.tolist()
    if not isinstance(nn_prediction, list):
        nn_prediction = [nn_prediction]

    flattened = []
    for value in nn_prediction:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    nn_prediction = [float(x) for x in flattened]
    print(f"NN values: {nn_prediction}")

    if len(nn_prediction) == 3:
        nn_predictions = {"24h": nn_prediction[0], "48h": nn_prediction[1], "72h": nn_prediction[2]}
    elif len(nn_prediction) == 1:
        print("\nWARNING: NN has only one output — using it for all horizons.")
        nn_predictions = {"24h": nn_prediction[0], "48h": nn_prediction[0], "72h": nn_prediction[0]}
    else:
        raise ValueError(f"Unexpected NN output size: {len(nn_prediction)}")

    # --------------------------------------------------------
    # Ensemble change -> ADD to current_aqi -> final AQI
    # --------------------------------------------------------
    results = {}
    cb_weight = ENSEMBLE_WEIGHTS["catboost"]
    nn_weight = ENSEMBLE_WEIGHTS["nn"]

    print("\nEnsemble predictions:")
    for horizon in HORIZONS:
        cb_pred = catboost_predictions[horizon]
        nn_pred = nn_predictions[horizon]

        predicted_change = cb_weight * cb_pred + nn_weight * nn_pred
        predicted_aqi = max(0.0, current_aqi + predicted_change)

        results[horizon] = {
            "catboost_prediction": round(float(cb_pred), 2),
            "nn_prediction": round(float(nn_pred), 2),
            "predicted_change": round(float(predicted_change), 2),
            "predicted_aqi": round(float(predicted_aqi), 1),
        }

        print(
            f"  {horizon}: change={predicted_change:+.2f} -> "
            f"AQI={predicted_aqi:.1f} (CatBoost={cb_pred:.2f}, NN={nn_pred:.2f})"
        )

    results["current_aqi"] = round(current_aqi, 1)
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("KARACHI AQI FORECAST")
    print("=" * 60)

    try:
        # 1. Check files
        check_required_files()

        # 2. Load feature columns
        feature_columns = load_feature_columns()

        # 3. Build LIVE features (local history + live Open-Meteo forecast)
        row_df, current_aqi = get_latest_features(feature_columns)

        # 4. Load models
        models = load_models()

        # 5. Predict
        forecast = predict_aqi(row_df, current_aqi, models, feature_columns)

        # 6. Print final results
        print("\n")
        print("=" * 60)
        print("FINAL AQI FORECAST")
        print("=" * 60)

        print(f"\nCurrent AQI: {forecast['current_aqi']}")

        for horizon in HORIZONS:
            result = forecast[horizon]
            print(f"\n{horizon} Forecast")
            print(f"  CatBoost : {result['catboost_prediction']}")
            print(f"  Neural Net: {result['nn_prediction']}")
            print(
                f"  Ensemble :   Ensemble AQI : {result['predicted_aqi']} "
                f"(change: {result['predicted_change']:+.2f})"
            )

        print("\n" + "=" * 60)
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print("\n")
        print("=" * 60)
        print("PREDICTION FAILED")
        print("=" * 60)
        print(f"\nError: {type(e).__name__}")
        print(f"Message: {e}")
        raise