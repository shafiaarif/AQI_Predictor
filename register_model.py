"""
register_model.py
------------------
FS70 Ensemble (CatBoost + Neural Network) model ko Hopsworks Model Registry
mein register karta hai.

IMPORTANT: saved_models/fs70/ mein sirf CatBoost + Neural Network nahi hain —
train_model.py ne Random Forest, Ridge, XGBoost ke models bhi usi folder mein
save kiye the (5 models per feature set). predict.py sirf CatBoost + NN use
karta hai, to upload se pehle sirf zaroori files ek clean temp folder mein
copy kar ke, wahi upload karte hain. Isse:
  1) upload size bohot kam ho jata hai (40MB+ ki Random Forest file skip)
  2) large-file upload ke dauran SSL/timeout error ka risk ghat jata hai

Run:
    python register_model.py
"""

import os
import shutil
import hopsworks

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "saved_models", "fs70")
DEPLOY_DIR = os.path.join(BASE_DIR, "saved_models", "fs70_deploy")

# training run se maloom metrics
METRICS = {
    "rmse": 6.5013,
    "mae": 5.0739,
    "r2": 0.4675,
}

# Sirf ye files/folders chahiye — baaki (random_forest, ridge, xgboost, etc.)
# skip ho jayenge.
REQUIRED_ITEMS = [
    "catboost",
    "neural_network",
    "feature_columns.json",
]

# ---------------------------------------------------------
# 2. Clean deploy folder banao (sirf zaroori files ke saath)
# ---------------------------------------------------------
print("=" * 60)
print("PREPARING CLEAN DEPLOY FOLDER")
print("=" * 60)

if os.path.exists(DEPLOY_DIR):
    shutil.rmtree(DEPLOY_DIR)
os.makedirs(DEPLOY_DIR)

for item in REQUIRED_ITEMS:
    src = os.path.join(SOURCE_DIR, item)

    if not os.path.exists(src):
        raise FileNotFoundError(f"Required item not found in source:\n{src}")

    dst = os.path.join(DEPLOY_DIR, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

    print(f"[COPIED] {item}")

# Total size check (sirf info ke liye)
total_size_mb = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(DEPLOY_DIR)
    for f in files
) / (1024 * 1024)

print(f"\nDeploy folder ready: {DEPLOY_DIR}")
print(f"Total upload size: {total_size_mb:.2f} MB\n")

# ---------------------------------------------------------
# 3. Sanity check
# ---------------------------------------------------------
REQUIRED_FILES = [
    os.path.join(DEPLOY_DIR, "catboost", "target_aqi_24.pkl"),
    os.path.join(DEPLOY_DIR, "catboost", "target_aqi_48.pkl"),
    os.path.join(DEPLOY_DIR, "catboost", "target_aqi_72.pkl"),
    os.path.join(DEPLOY_DIR, "neural_network", "model.keras"),
    os.path.join(DEPLOY_DIR, "neural_network", "scaler.pkl"),
    os.path.join(DEPLOY_DIR, "feature_columns.json"),
]

print("=" * 60)
print("CHECKING DEPLOY FILES")
print("=" * 60)

for path in REQUIRED_FILES:
    if os.path.exists(path):
        print(f"[OK] {path}")
    else:
        raise FileNotFoundError(f"Required file not found:\n{path}")

print("All required model files found!\n")

# ---------------------------------------------------------
# 4. Connect to Hopsworks
# ---------------------------------------------------------
print("=" * 60)
print("CONNECTING TO HOPSWORKS")
print("=" * 60)

project = hopsworks.login()
mr = project.get_model_registry()
print(f"Connected to project: {project.name}\n")

# ---------------------------------------------------------
# 5. Register model (sirf clean deploy folder upload hoga)
# ---------------------------------------------------------
print("=" * 60)
print("REGISTERING MODEL")
print("=" * 60)

aqi_model = mr.python.create_model(
    name="aqi_predictor_fs70_ensemble",
    metrics=METRICS,
    description=(
        "Karachi AQI forecasting model - FS70 feature set, "
        "CatBoost + Neural Network ensemble. Predicts AQI CHANGE for "
        "24h/48h/72h horizons; final AQI = current_us_aqi + predicted_change. "
        "Folder includes: catboost/*.pkl, neural_network/model.keras + scaler.pkl, "
        "feature_columns.json. (Random Forest / Ridge / XGBoost excluded — not used "
        "by the deployed ensemble.)"
    ),
)

aqi_model.save(DEPLOY_DIR)

print(f"\nModel registered successfully: {aqi_model.name} v{aqi_model.version}")
print(f"View it at: {project.get_url()}")