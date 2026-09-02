"""
feature_view.py

Creates (or fetches, if already created) the Hopsworks Feature View on top
of the karachi_aqi_features Feature Group.

IMPORTANT — VERSION LOCKSTEP:
    FEATURE_GROUP_VERSION here MUST match FEATURE_GROUP_VERSION in
    feature_store.py (the script that writes the group) AND in
    train_model.py (if you ever switch train_model.py to read from the
    view instead of the group directly). Whenever feature_engineering.py's
    schema changes and you bump the version in feature_store.py, bump it
    here too, in the same commit.

WHAT THIS BUILDS:
    A Feature View selecting every column from the Feature Group EXCEPT
    target_change_24/48/72. Those auxiliary "change" targets stay in the
    underlying Feature Group (train_model.py reads them directly from
    there), but they are intentionally left out of this View's label set
    to avoid two overlapping definitions of "the target" for anything
    that consumes this View downstream (e.g. a future predict.py/dashboard
    doing inference, or an alternate training path that uses
    feature_view.training_data() instead of a raw group read).

    Labels for this View are the three absolute AQI targets:
        target_aqi_24, target_aqi_48, target_aqi_72

USAGE NOTES:
    - Safe to re-run: get_or_create_feature_view() returns the existing
      view if one with this exact name+version+query already exists, and
      raises if you try to reuse the version with a DIFFERENT query
      (Hopsworks feature views are immutable once created) — bump
      FEATURE_VIEW_VERSION if you need to change the query/labels later.
    - train_model.py does NOT currently use this view (it reads the
      Feature Group directly). This is fine for training as-is; wire this
      view in wherever you build predict.py / the Streamlit dashboard, or
      if you switch train_model.py to feature_view.get_batch_data() /
      training_data() splits instead of manual time-based slicing.
"""

import os
import hopsworks

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional locally; in CI the env vars are injected directly
    pass


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 10  # MUST match feature_store.py's FEATURE_GROUP_VERSION

FEATURE_VIEW_NAME = "karachi_aqi_feature_view"
FEATURE_VIEW_VERSION = 10  # keep in lockstep with FEATURE_GROUP_VERSION

LABEL_COLUMNS = [
    "target_aqi_24",
    "target_aqi_48",
    "target_aqi_72",
]

# Excluded from the view entirely (still present in the underlying Feature
# Group — train_model.py reads them from there directly for its
# change-target training approach).
EXCLUDED_COLUMNS = [
    "target_change_24",
    "target_change_48",
    "target_change_72",
]


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 1. CONNECT TO HOPSWORKS
# ============================================================

print_section("CONNECTING TO HOPSWORKS")

api_key = os.environ.get("HOPSWORKS_API_KEY")
project_name = os.environ.get("HOPSWORKS_PROJECT")

if not api_key:
    raise EnvironmentError(
        "HOPSWORKS_API_KEY environment variable not set. "
        "Add it to a local .env file or as a GitHub Actions secret."
    )

if not project_name:
    raise EnvironmentError(
        "HOPSWORKS_PROJECT environment variable not set. "
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


# ============================================================
# 2. GET FEATURE STORE + FEATURE GROUP
# ============================================================

fs = project.get_feature_store()
print("Feature Store accessed successfully!")

print_section("FETCHING FEATURE GROUP")

feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
)

print("Feature Group:", feature_group.name, "| Version:", feature_group.version)


# ============================================================
# 3. BUILD QUERY (everything except the auxiliary change targets)
# ============================================================

print_section("BUILDING QUERY")

query = feature_group.select_except(EXCLUDED_COLUMNS)

print("Excluded columns:", EXCLUDED_COLUMNS)
print("Label columns   :", LABEL_COLUMNS)


# ============================================================
# 4. CREATE / GET FEATURE VIEW
# ============================================================

print_section("CREATING FEATURE VIEW")

feature_view = fs.get_or_create_feature_view(
    name=FEATURE_VIEW_NAME,
    version=FEATURE_VIEW_VERSION,
    query=query,
    labels=LABEL_COLUMNS,
    description=(
        # Hopsworks caps descriptions at 256 chars, same as the
        # Feature Group — keep this short if you edit it.
        "Karachi AQI feature view (V4 schema, forecast-lead features "
        "included). Labels: target_aqi_24/48/72. Excludes "
        "target_change_* (kept in the Feature Group only)."
    ),
)

print("Feature View:", feature_view.name, "| Version:", feature_view.version)


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

print_section("FEATURE VIEW SETUP COMPLETED")

print("Feature Group:", feature_group.name, "| Version:", feature_group.version)
print("Feature View :", feature_view.name, "| Version:", feature_view.version)
print("Labels       :", LABEL_COLUMNS)
print("Excluded     :", EXCLUDED_COLUMNS)
print("\nReady for training-dataset creation or batch inference.")
print("=" * 60)