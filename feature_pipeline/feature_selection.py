"""
feature_selection.py
============================================================

Karachi AQI Forecasting - Multi-Feature-Set Selection

PURPOSE
-------
This script automatically generates 5 different feature sets so that
we can later compare model performance with different numbers of
selected features.

PIPELINE
--------
1. Load engineered feature dataset.
2. Exclude target columns and time.
3. Use only the training-like portion of the data.
   - Last 90 days are excluded.
   - This prevents validation/test leakage during feature selection.
4. Correlation pruning:
   - If two features have correlation > 0.95,
     keep the more relevant feature.
5. Random Forest importance:
   - Train one RF per AQI change horizon.
   - Select TOP_N features for each horizon.
6. Union the selected features across 24h/48h/72h.
7. Generate 5 feature-set JSON files.

FEATURE-SET TARGETS
-------------------
TOP_N_VALUES = [50, 70, 90, 110, 130]

IMPORTANT
---------
TOP_N is applied PER HORIZON.

Because features are unioned across:
    24h + 48h + 72h

the final number of features will usually be larger than TOP_N.

OUTPUT
------
data/processed/feature_selection_sets/

    feature_columns_selected_50.json
    feature_columns_selected_70.json
    feature_columns_selected_90.json
    feature_columns_selected_110.json
    feature_columns_selected_130.json

Also creates:

    feature_selection_summary.json

The existing:
    feature_columns_selected.json

is NOT overwritten by default.

This protects your current 91-feature experiment.

Run:
    python feature_selection.py
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "karachi_features_v3.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "feature_selection_sets"
)

SUMMARY_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "feature_selection_summary.json"
)

# Target columns
TARGET_COLUMNS = [
    "target_aqi_24",
    "target_aqi_48",
    "target_aqi_72",
]

CHANGE_TARGET_COLUMNS = [
    "target_change_24",
    "target_change_48",
    "target_change_72",
]

# Correlation threshold
CORRELATION_THRESHOLD = 0.95

# Five feature-selection experiments
TOP_N_VALUES = [50, 70, 90, 110, 130]

# Random Forest used ONLY for feature importance
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 4

RANDOM_STATE = 42

# Exclude last 90 days from feature selection
# This corresponds to:
#     VAL_DAYS = 30
#     TEST_DAYS = 60
SELECTION_EXCLUSION_DAYS = 90


# ============================================================
# HELPER
# ============================================================

def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print_section("[1] LOADING FEATURE DATASET")

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        f"\nFeature file not found:\n{FEATURE_FILE}\n\n"
        "Run feature_engineering.py first."
    )

df = pd.read_csv(FEATURE_FILE)

if "time" not in df.columns:
    raise ValueError("Dataset does not contain required 'time' column.")

df["time"] = pd.to_datetime(df["time"])

df = (
    df
    .sort_values("time")
    .reset_index(drop=True)
)

print(f"Feature file : {FEATURE_FILE}")
print(f"Rows         : {len(df)}")
print(f"Columns      : {len(df.columns)}")
print(f"Date range   : {df['time'].min()} → {df['time'].max()}")


# ============================================================
# 2. CHECK REQUIRED TARGETS
# ============================================================

required_columns = (
    TARGET_COLUMNS
    + CHANGE_TARGET_COLUMNS
    + ["time"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns:\n{missing_columns}"
    )


# ============================================================
# 3. CANDIDATE FEATURES
# ============================================================

print_section("[2] PREPARING CANDIDATE FEATURES")

excluded_columns = (
    TARGET_COLUMNS
    + CHANGE_TARGET_COLUMNS
    + ["time"]
)

candidate_features = [
    col
    for col in df.columns
    if col not in excluded_columns
]

print(
    f"Candidate features before selection: "
    f"{len(candidate_features)}"
)

print(
    "Excluded columns:",
    excluded_columns
)


# ============================================================
# 4. TRAINING-LIKE DATA ONLY
# ============================================================

print_section("[3] CREATING LEAKAGE-SAFE SELECTION DATA")

selection_cutoff = (
    df["time"].max()
    - pd.Timedelta(days=SELECTION_EXCLUSION_DAYS)
)

selection_df = (
    df[df["time"] < selection_cutoff]
    .reset_index(drop=True)
)

print(
    f"Selection cutoff : {selection_cutoff}"
)

print(
    f"Rows used        : {len(selection_df)}"
)

print(
    f"Rows excluded    : {len(df) - len(selection_df)}"
)

print(
    "Excluded period  : last "
    f"{SELECTION_EXCLUSION_DAYS} days"
)

if len(selection_df) < 1000:
    raise ValueError(
        "Too few rows available for feature selection."
    )


# ============================================================
# 5. HANDLE NUMERIC FEATURES
# ============================================================

print_section("[4] VALIDATING FEATURE TYPES")

non_numeric_features = []

for feature in candidate_features:
    if not pd.api.types.is_numeric_dtype(
        selection_df[feature]
    ):
        non_numeric_features.append(feature)

if non_numeric_features:

    print(
        "WARNING: Non-numeric features detected:"
    )

    for feature in non_numeric_features:
        print(f"  - {feature}")

    print(
        "\nThese features will be excluded from "
        "feature selection."
    )

    candidate_features = [
        f
        for f in candidate_features
        if f not in non_numeric_features
    ]

print(
    f"Numeric candidate features: "
    f"{len(candidate_features)}"
)


# ============================================================
# 6. CORRELATION PRUNING
# ============================================================

print_section("[5] CORRELATION PRUNING")

X = selection_df[candidate_features].copy()

print(
    "Computing absolute correlation matrix..."
)

corr_matrix = X.corr().abs()

print("Correlation matrix ready.")


# ------------------------------------------------------------
# Relevance score
# ------------------------------------------------------------
#
# We use average absolute correlation with the three
# CHANGE targets as a quick relevance measure.
#
# This is NOT the final importance ranking.
# The Random Forest ranking comes later.
#

avg_change_target = (
    selection_df[CHANGE_TARGET_COLUMNS]
    .mean(axis=1)
)

relevance = X.apply(
    lambda col: abs(
        col.corr(avg_change_target)
    )
)

relevance = relevance.fillna(0)

ordered_features = (
    relevance
    .sort_values(ascending=False)
    .index
    .tolist()
)


# ------------------------------------------------------------
# Keep one feature from highly-correlated groups
# ------------------------------------------------------------

kept_features = []
dropped_features = []
dropped_reason = {}

for feature in ordered_features:

    is_redundant = False

    for kept in kept_features:

        correlation = corr_matrix.loc[
            feature,
            kept
        ]

        if correlation > CORRELATION_THRESHOLD:

            is_redundant = True

            dropped_features.append(feature)

            dropped_reason[feature] = (
                f"correlated {correlation:.3f} "
                f"with '{kept}'"
            )

            break

    if not is_redundant:
        kept_features.append(feature)


print(
    f"\nFeatures before pruning : "
    f"{len(candidate_features)}"
)

print(
    f"Features after pruning  : "
    f"{len(kept_features)}"
)

print(
    f"Dropped as redundant    : "
    f"{len(dropped_features)}"
)


# ------------------------------------------------------------
# Show examples
# ------------------------------------------------------------

if dropped_features:

    print(
        "\nExample redundant features:"
    )

    for feature in dropped_features[:20]:

        print(
            f"  - {feature}"
            f"  ({dropped_reason[feature]})"
        )


# ============================================================
# 7. RANDOM FOREST IMPORTANCE
# ============================================================

print_section(
    "[6] RANDOM FOREST FEATURE IMPORTANCE"
)

X_pruned = (
    selection_df[kept_features]
    .copy()
)

selected_features_by_horizon = {}

importance_tables = {}


for change_col in CHANGE_TARGET_COLUMNS:

    print(
        f"\nTraining importance model for "
        f"{change_col}..."
    )

    y = selection_df[change_col]

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    rf.fit(
        X_pruned,
        y
    )

    importances = pd.Series(
        rf.feature_importances_,
        index=kept_features
    )

    importances = (
        importances
        .sort_values(ascending=False)
    )

    importance_tables[change_col] = importances

    print(
        f"Importance model trained for "
        f"{change_col}"
    )

    print("\nTop 15 features:")

    for feature, importance in (
        importances.head(15).items()
    ):

        print(
            f"  {feature:<50}"
            f"{importance:.6f}"
        )


# ============================================================
# 8. GENERATE FIVE FEATURE SETS
# ============================================================

print_section(
    "[7] GENERATING FIVE FEATURE SETS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

feature_set_summary = {}


for top_n in TOP_N_VALUES:

    print("\n" + "-" * 75)

    print(
        f"GENERATING TOP_N = {top_n}"
    )

    print("-" * 75)

    selected_per_horizon = {}

    # --------------------------------------------------------
    # Select top N independently for each horizon
    # --------------------------------------------------------

    for change_col in CHANGE_TARGET_COLUMNS:

        importances = (
            importance_tables[change_col]
        )

        top_features = (
            importances
            .head(top_n)
            .index
            .tolist()
        )

        selected_per_horizon[
            change_col
        ] = set(top_features)

        print(
            f"{change_col}: "
            f"{len(top_features)} features"
        )


    # --------------------------------------------------------
    # Union across horizons
    # --------------------------------------------------------

    final_selected = set()

    for features in selected_per_horizon.values():

        final_selected.update(features)

    final_selected = sorted(
        final_selected
    )

    actual_count = len(final_selected)

    print(
        f"\nFinal UNION feature count: "
        f"{actual_count}"
    )

    print(
        f"Requested TOP_N per horizon: "
        f"{top_n}"
    )


    # ========================================================
    # Feature breakdown
    # ========================================================

    forecast_count = len([
        f
        for f in final_selected
        if f.startswith("forecast_")
    ])

    lag_count = len([
        f
        for f in final_selected
        if f.startswith("aqi_lag_")
    ])

    rolling_count = len([
        f
        for f in final_selected
        if f.startswith("aqi_rolling_")
    ])

    other_count = (
        actual_count
        - forecast_count
        - lag_count
        - rolling_count
    )

    print(
        "\nFeature breakdown:"
    )

    print(
        f"  forecast weather : "
        f"{forecast_count}"
    )

    print(
        f"  AQI lags         : "
        f"{lag_count}"
    )

    print(
        f"  AQI rolling      : "
        f"{rolling_count}"
    )

    print(
        f"  other            : "
        f"{other_count}"
    )


    # ========================================================
    # Save JSON
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        f"feature_columns_selected_{top_n}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_selected,
            f,
            indent=2
        )

    print(
        f"\nSaved:"
        f"\n{output_file}"
    )


    # ========================================================
    # Save metadata
    # ========================================================

    feature_set_summary[
        str(top_n)
    ] = {
        "requested_top_n_per_horizon": top_n,
        "actual_union_feature_count": actual_count,

        "horizon_feature_counts": {
            change_col: len(
                selected_per_horizon[
                    change_col
                ]
            )
            for change_col in CHANGE_TARGET_COLUMNS
        },

        "forecast_weather_features":
            forecast_count,

        "aqi_lag_features":
            lag_count,

        "aqi_rolling_features":
            rolling_count,

        "other_features":
            other_count,

        "output_file":
            output_file,

        "features":
            final_selected,
    }


# ============================================================
# 9. SAVE SUMMARY
# ============================================================

print_section(
    "[8] SAVING FEATURE SELECTION SUMMARY"
)

summary = {

    "dataset": FEATURE_FILE,

    "candidate_feature_count":
        len(candidate_features),

    "features_after_correlation_pruning":
        len(kept_features),

    "correlation_threshold":
        CORRELATION_THRESHOLD,

    "selection_exclusion_days":
        SELECTION_EXCLUSION_DAYS,

    "selection_cutoff":
        str(selection_cutoff),

    "top_n_values":
        TOP_N_VALUES,

    "random_forest": {
        "n_estimators":
            RF_N_ESTIMATORS,

        "max_depth":
            RF_MAX_DEPTH,

        "min_samples_leaf":
            RF_MIN_SAMPLES_LEAF,

        "random_state":
            RANDOM_STATE,
    },

    "feature_sets":
        feature_set_summary,
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print(
    f"Saved summary:\n{SUMMARY_FILE}"
)


# ============================================================
# 10. FINAL SUMMARY TABLE
# ============================================================

print_section(
    "[9] FINAL FEATURE SET SUMMARY"
)

print(
    f"{'TOP_N':>8}"
    f"{'FINAL FEATURES':>18}"
    f"{'FORECAST':>12}"
    f"{'LAGS':>10}"
    f"{'ROLLING':>12}"
    f"{'OTHER':>10}"
)

print("-" * 72)


for top_n in TOP_N_VALUES:

    info = feature_set_summary[
        str(top_n)
    ]

    print(
        f"{top_n:>8}"
        f"{info['actual_union_feature_count']:>18}"
        f"{info['forecast_weather_features']:>12}"
        f"{info['aqi_lag_features']:>10}"
        f"{info['aqi_rolling_features']:>12}"
        f"{info['other_features']:>10}"
    )


# ============================================================
# 11. IMPORTANT NOTES
# ============================================================

print_section(
    "[10] COMPLETE"
)

print(
    "Five feature sets have been generated."
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nGenerated files:"
)

for top_n in TOP_N_VALUES:

    print(
        f"  feature_columns_selected_{top_n}.json"
    )

print(
    "\nSummary:"
)

print(
    "  feature_selection_summary.json"
)

print(
    "\nIMPORTANT:"
)

print(
    "TOP_N is applied separately to 24h, 48h and 72h."
)

print(
    "The final feature set is the UNION across all three horizons."
)

print(
    "Therefore the actual number of features can be greater"
)

print(
    "than TOP_N."
)

print(
    "\nYour existing feature_columns_selected.json"
)

print(
    "was NOT overwritten."
)

print(
    "\nNext step:"
)

print(
    "Modify train_model.py to run these five feature sets"
)

print(
    "and compare their validation/test performance."
)

print("=" * 75)