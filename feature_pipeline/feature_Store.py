"""
feature_store.py

Uploads the engineered feature dataset to the Hopsworks Feature Store.

AUTH:
Instead of a hardcoded local cert folder path (which only works on your
Windows machine), this uses an API key from an environment variable, so it
works both locally and inside GitHub Actions.

Local setup:
    1. Get your API key from the Hopsworks UI (Account Settings -> API Keys)
    2. Create a .env file in your project root with:
           HOPSWORKS_API_KEY=your_key_here
           HOPSWORKS_PROJECT=your_project_name
    3. pip install python-dotenv hopsworks

GitHub Actions setup:
    Add HOPSWORKS_API_KEY and HOPSWORKS_PROJECT as repo secrets, then pass
    them as env vars to this step in your workflow yaml.
"""

import os
import pandas as pd
import hopsworks

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional locally; in CI the env vars are injected directly
    pass


# __file__-based paths so this works no matter which directory you run it
# from (matches feature_engineering.py's path resolution).
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FEATURE_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "karachi_features_v3.csv")

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 10 # bumped: V4 feature set (188 features, incl. 36 forecast-lead
                          # weather columns added on top of the prior V3 schema) has a
                          # different schema and must not overwrite the old version.


# ==================================================
# 1. LOAD PROCESSED FEATURE DATASET
# ==================================================

print("\n" + "=" * 60)
print("LOADING FEATURE DATASET")
print("=" * 60)

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        f"{FEATURE_FILE} not found. Run feature_engineering.py first."
    )

df = pd.read_csv(FEATURE_FILE)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Time dtype:", df["time"].dtype)


# ==================================================
# 2. CHECK TARGET COLUMNS
# ==================================================

target_columns = ["target_aqi_24", "target_aqi_48", "target_aqi_72"]

missing_targets = [t for t in target_columns if t not in df.columns]

if missing_targets:
    raise ValueError(f"Missing target columns: {missing_targets}")

print("\nAll target columns found successfully!")


# ==================================================
# 3. BASIC DATA VALIDATION
# ==================================================

print("\n" + "=" * 60)
print("DATA VALIDATION")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", df.isnull().sum().sum())
print("Duplicate timestamps:", df["time"].duplicated().sum())

is_sorted = df["time"].is_monotonic_increasing
print("Chronological order:", is_sorted)

if not is_sorted:
    raise ValueError("Dataset is not chronologically sorted.")

if df.isnull().sum().sum() != 0:
    raise ValueError("Dataset contains missing values.")

if df["time"].duplicated().sum() != 0:
    raise ValueError("Duplicate timestamps found.")

print("Data validation passed!")


# ==================================================
# 4. CONNECT TO HOPSWORKS
# ==================================================

print("\n" + "=" * 60)
print("CONNECTING TO HOPSWORKS")
print("=" * 60)

api_key = os.environ.get("HOPSWORKS_API_KEY")
project_name = os.environ.get("HOPSWORKS_PROJECT")

if not api_key:
    raise EnvironmentError(
        "HOPSWORKS_API_KEY environment variable not set. "
        "Add it to a local .env file or as a GitHub Actions secret."
    )


project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    project=project_name,
    api_key_value=api_key,
)

print("Connected to Hopsworks!")
print("Project:", project.name)


# ==================================================
# 5. GET FEATURE STORE
# ==================================================

fs = project.get_feature_store()
print("Feature Store accessed successfully!")


# ==================================================
# 6. CREATE / GET FEATURE GROUP
# ==================================================

print("\n" + "=" * 60)
print("CREATING FEATURE GROUP")
print("=" * 60)

feature_group = fs.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description=(
        # NOTE: Hopsworks caps feature-group descriptions at 256 characters
        # (a longer one raises errorCode 270092 / HTTP 400 on insert) — keep
        # this short if you ever edit it.
        "Hourly Karachi AQI features V4: lags, rolling stats, "
        "trend/deviation, same-hour history, pollutant ratios, weather "
        "interactions, wind components, plus 36 forecast-lead weather "
        "features (t+24/48/72h ahead). For 24h/48h/72h AQI prediction."
    ),
    primary_key=["time"],
    event_time="time",
    online_enabled=False,
    time_travel_format="HUDI",
)

print("Feature Group created/accessed successfully!")
print("Feature Group:", feature_group.name)
print("Version:", feature_group.version)


# ==================================================
# 7. INSERT DATA
# ==================================================

print("\n" + "=" * 60)
print("INSERTING DATA")
print("=" * 60)

feature_group.insert(df)

print("\nData inserted successfully!")
print("Rows inserted:", len(df))


# ==================================================
# 8. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FEATURE STORE SETUP COMPLETED")
print("=" * 60)

print("Feature Group:", feature_group.name, "| Version:", feature_group.version)
print("Rows:", len(df), "| Columns:", len(df.columns))
print("Targets:", target_columns)
print("Date range:", df["time"].min(), "→", df["time"].max())
print("\nReady for Feature View creation!")
print("=" * 60)