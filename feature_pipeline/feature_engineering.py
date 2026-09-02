"""
feature_engineering.py (V4 - FORECAST-LEAD FEATURES ADDED)

Karachi AQI Forecasting Feature Engineering Pipeline

INPUT:
    data/raw_dataset/karachi_processed.csv

OUTPUT:
    data/processed/karachi_features_v3.csv
    data/processed/feature_columns_v3.json

MAIN TARGETS (all three are trained on — see train_model.py):
    target_aqi_24
    target_aqi_48
    target_aqi_72

AUXILIARY TARGETS (kept in the dataset for analysis, NEVER used as inputs):
    target_change_24
    target_change_48
    target_change_72

============================================================
NEW IN V4: FORECAST-LEAD WEATHER FEATURES
============================================================
Previously, every input feature only used information from time t or
earlier (lags, rolling stats, etc.). This meant the model had to predict
AQI 24h/48h/72h ahead with ZERO knowledge of what the weather would
actually be doing during that window — despite wind, rain, and humidity
being the dominant physical drivers of pollutant dispersion.

This version adds "forecast-lead" features: the ACTUAL weather value at
t+24h, t+48h, and t+72h, built with a negative shift (df[col].shift(-h))
on the historical weather columns.

WHY THIS IS NOT LEAKAGE:
At real-time inference, this exact information is available from the
Open-Meteo FORECAST API (already used in fetch_data.py's
FORECAST_WEATHER_URL) — a 16-day rolling weather forecast is public and
known in advance. During TRAINING, we don't have live forecast API calls
for 2024-2026 history, so we substitute the historical ACTUAL weather
that occurred at t+h as a stand-in for "what the forecast would have
said" (forecasts for short lead times like 24-72h are typically close to
the eventual actual value for this kind of data). This is standard
practice for training forecast-driven pipelines offline. The important
distinction: unlike target_aqi_* (which is derived from us_aqi — the
quantity we're trying to predict), forecast-lead weather columns are an
INDEPENDENT input signal (wind/rain/pressure), not a disguised copy of
the label, so using them as features is legitimate.

At inference time (predict.py / the dashboard), these forecast_* columns
must be populated from a LIVE call to the Open-Meteo forecast endpoint,
not from historical data (which won't exist yet for future timestamps).

IMPORTANT:
    - All model input features use information available at time t or earlier
      OR information about the future that is knowable in advance via a
      weather forecast API (the new forecast_* columns below).
    - No future AQI is used as an input feature.
    - Future AQI is ONLY stored in target columns.
    - target_* columns MUST NOT be used as model input (verified in [22]).

PERFORMANCE NOTE (fix from the previous version):
    Instead of inserting ~150 new columns into `df` one at a time
    (df["new_col"] = ...), which pandas has to re-consolidate memory for on
    every single assignment and which triggers "DataFrame is highly
    fragmented" warnings, every new feature is first written into a plain
    Python dict (`new_columns`). All new columns are then attached to `df`
    in ONE pd.concat call. Same output, much faster, no warnings.

============================================================
FIXES IN THIS VERSION (post-preprocess.py run)
============================================================
Running preprocess.py's output through the original script surfaced two
real problems that would have crashed the pipeline or silently destroyed
the dataset:

1. HOURLY CONTINUITY: the raw feed had 1 non-hourly gap. The original
   script hard-raised on any gap, stopping the pipeline dead. This
   version instead reindexes the dataframe onto a complete hourly
   DatetimeIndex first, inserting a NaN row for any missing hour. The
   continuity check then legitimately passes, and the single inserted
   NaN row is removed later at the normal dropna() cleaning step — same
   place every other NaN-causing row (from lags/targets) already gets
   removed, so no special-casing is needed downstream.

2. UNUSED, MOSTLY-NULL RAW COLUMNS (e.g. "visibility"): the raw dataset
   carries columns that are never referenced by REQUIRED_COLUMNS or
   FORECAST_LEAD_COLUMNS, but they still rode along in `df` all the way
   to the final blanket df.dropna() call. When such a column is almost
   entirely NaN (as "visibility" was — ~99.8% missing), that one
   dropna() call wipes out almost every row in the dataset, silently
   shrinking tens of thousands of rows down to a couple hundred with NO
   warning or error. This version now restricts `df` to REQUIRED_COLUMNS
   only right after the required-columns check, so any such dead-weight
   column is dropped before it can contaminate the final dataset.
"""


import os
import json
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw_dataset")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_FILE = os.path.join(RAW_DIR, "karachi_processed.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "karachi_features_v3.csv")
FEATURE_LIST_FILE = os.path.join(PROCESSED_DIR, "feature_columns_v3.json")


# ============================================================
# 2. CONFIGURATION
# ============================================================

LAG_HOURS = [1, 3, 6, 12, 24, 48, 72, 96, 120, 168, 336, 504]
ROLLING_WINDOWS = [6, 12, 24, 48, 72, 168]
CHANGE_WINDOWS = [1, 3, 6, 12, 24, 48, 72]
PCT_CHANGE_WINDOWS = [3, 6, 12, 24, 48, 72, 168]
TREND_WINDOWS = [6, 12, 24, 48, 72, 168]
DEVIATION_WINDOWS = [24, 48, 72, 168]   # must be a subset of ROLLING_WINDOWS
TARGET_HOURS = [24, 48, 72]

# NEW (V4): which weather/pollutant columns get a forward-looking
# "forecast-lead" feature, and at which horizons. These are the columns
# that physically drive pollutant dispersion (wind disperses/concentrates
# pollutants, rain washes them out, humidity/pressure affect chemistry),
# and — critically — are all available from a real weather forecast API at
# inference time (unlike AQI itself, which open-meteo doesn't forecast for
# us the way we need).
FORECAST_LEAD_HOURS = [24, 48, 72]
FORECAST_LEAD_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
]


# ============================================================
# 3. REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "time", "us_aqi",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "cloud_cover", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "weather_code",
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "european_aqi",
]


# ============================================================
# 4. START
# ============================================================

print("\n" + "=" * 70)
print("KARACHI AQI FEATURE ENGINEERING V4 (forecast-lead features added)")
print("=" * 70)
print("\nInput file :", INPUT_FILE)
print("Output file:", OUTPUT_FILE)


# ============================================================
# 5. CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\n\nProcessed dataset NOT FOUND!\nExpected file:\n{INPUT_FILE}\n\n"
        "Run preprocess.py first."
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("[1] LOADING PREPROCESSED DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("Rows   :", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("[2] CHECKING REQUIRED COLUMNS")
print("=" * 70)

missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing_columns:
    print("\nMissing columns:")
    for c in missing_columns:
        print(" -", c)
    raise ValueError("\nRequired columns are missing. Check preprocess.py output.")

missing_forecast_source_columns = [c for c in FORECAST_LEAD_COLUMNS if c not in df.columns]
if missing_forecast_source_columns:
    print("\nMissing columns needed for forecast-lead features:")
    for c in missing_forecast_source_columns:
        print(" -", c)
    raise ValueError(
        "\nFORECAST_LEAD_COLUMNS references columns not present in the "
        "processed dataset. Check preprocess.py output or update "
        "FORECAST_LEAD_COLUMNS."
    )

print("All required columns found.")

# --------------------------------------------------------------
# FIX #2: drop any raw column not in REQUIRED_COLUMNS.
#
# The raw feed can carry extra columns (e.g. "visibility") that are
# never used by REQUIRED_COLUMNS or FORECAST_LEAD_COLUMNS. If such a
# column is left in `df`, it survives all the way to the final blanket
# df.dropna() call. When that column is mostly NaN (visibility was
# ~99.8% missing in this dataset), that single dropna() wipes out
# almost every row with no warning. Restricting to REQUIRED_COLUMNS
# here — before any feature engineering — prevents that.
# --------------------------------------------------------------

extra_columns = [c for c in df.columns if c not in REQUIRED_COLUMNS]

if extra_columns:
    print("\nDropping unused raw columns not in REQUIRED_COLUMNS:")
    for c in extra_columns:
        null_pct = 100 * df[c].isna().mean()
        print(f" - {c} ({null_pct:.1f}% missing)")
    df = df.drop(columns=extra_columns)

print("\nColumns kept for feature engineering:", len(df.columns))


# ============================================================
# 8. PARSE TIMESTAMP & SORT
# ============================================================

print("\nParsing timestamps...")

df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")

invalid_timestamps = int(df["time"].isna().sum())
if invalid_timestamps > 0:
    raise ValueError(f"{invalid_timestamps} invalid timestamps found.")

print("Timestamp parsing: PASS")

df = df.sort_values("time").reset_index(drop=True)
print("Start:", df["time"].min())
print("End  :", df["time"].max())


# ============================================================
# 9. DUPLICATE TIMESTAMP CHECK
# ============================================================

print("\n" + "=" * 70)
print("[3] CHECKING DUPLICATE TIMESTAMPS")
print("=" * 70)

duplicate_count = int(df["time"].duplicated().sum())
print("Duplicate timestamps:", duplicate_count)

if duplicate_count > 0:
    raise ValueError(f"\nFound {duplicate_count} duplicate timestamps.")

print("Duplicate check: PASS")


# ============================================================
# 10. HOURLY CONTINUITY CHECK (with auto-fill)
# ============================================================

print("\n" + "=" * 70)
print("[4] CHECKING HOURLY CONTINUITY")
print("=" * 70)

time_diff = df["time"].diff().dropna()
non_hourly = (time_diff != pd.Timedelta(hours=1))
non_hourly_count = int(non_hourly.sum())

print("Non-hourly intervals:", non_hourly_count)

# --------------------------------------------------------------
# FIX #1: instead of hard-raising on any gap, reindex onto a complete
# hourly range. Missing hours become NaN rows, which are removed later
# by the normal dropna() cleaning step (step 28/[19]) — the same place
# lag/target NaNs already get removed — so no special handling is
# needed further down the pipeline.
# --------------------------------------------------------------

if non_hourly_count > 0:
    problematic_indices = np.where(non_hourly.values)[0] + 1
    print("\nGap(s) found before these timestamps:")
    print(df.loc[problematic_indices[:10], "time"])

    full_range = pd.date_range(
        start=df["time"].min(),
        end=df["time"].max(),
        freq="h",
    )

    rows_before_reindex = len(df)

    df = (
        df.set_index("time")
        .reindex(full_range)
        .rename_axis("time")
        .reset_index()
    )

    rows_inserted = len(df) - rows_before_reindex

    print(
        f"\nReindexed onto a complete hourly range: "
        f"{rows_inserted} missing hour(s) inserted as NaN rows."
    )
    print(
        "These NaN rows will be removed automatically at the "
        "dropna() cleaning step later in the pipeline."
    )

    # Re-check to confirm the fix actually closed every gap.
    time_diff = df["time"].diff().dropna()
    non_hourly_count = int((time_diff != pd.Timedelta(hours=1)).sum())

print("Hourly continuity:", "PASS" if non_hourly_count == 0 else "FAIL")

if non_hourly_count > 0:
    raise ValueError(
        "\nDataset is still not continuously hourly after reindexing.\n"
        "This should not happen — investigate the timestamp data."
    )


# ============================================================
# 11. CHECK NUMERIC INPUT COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("[5] CHECKING NUMERIC COLUMNS")
print("=" * 70)

numeric_input_columns = [c for c in REQUIRED_COLUMNS if c != "time"]

for c in numeric_input_columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

numeric_missing = df[numeric_input_columns].isna().sum()
numeric_missing = numeric_missing[numeric_missing > 0]

if len(numeric_missing) > 0:
    print("\nNumeric columns containing missing values:")
    print(numeric_missing)
else:
    print("Numeric input check: PASS")


# ============================================================
# THE FIX: all engineered features go into this dict first.
# We attach everything to `df` in ONE concat at the end of each
# major stage that other stages depend on, to avoid fragmentation
# while still letting later formulas reference earlier results.
# ============================================================

new_columns = {}


# ============================================================
# 12. BASIC + CYCLICAL TIME FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[6] CREATING TIME FEATURES")
print("=" * 70)

hour = df["time"].dt.hour
day = df["time"].dt.day
month = df["time"].dt.month
day_of_week = df["time"].dt.dayofweek
day_of_year = df["time"].dt.dayofyear
week_of_year = df["time"].dt.isocalendar().week.astype(int)
is_weekend = (day_of_week >= 5).astype(int)

new_columns.update({
    "hour": hour,
    "day": day,
    "month": month,
    "day_of_week": day_of_week,
    "day_of_year": day_of_year,
    "week_of_year": week_of_year,
    "is_weekend": is_weekend,
    "hour_sin": np.sin(2 * np.pi * hour / 24),
    "hour_cos": np.cos(2 * np.pi * hour / 24),
    "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7),
    "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7),
    "month_sin": np.sin(2 * np.pi * month / 12),
    "month_cos": np.cos(2 * np.pi * month / 12),
    "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
    "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
})

print("Time features created:", 15)


# ============================================================
# 13. AQI LAG FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[7] CREATING AQI LAG FEATURES")
print("=" * 70)

for lag in LAG_HOURS:
    new_columns[f"aqi_lag_{lag}"] = df["us_aqi"].shift(lag)

print("Created", len(LAG_HOURS), "AQI lag features.")


# ============================================================
# 14. ROLLING AQI STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("[8] CREATING ROLLING AQI FEATURES")
print("=" * 70)

rolling_means = {}   # kept in a plain dict for reuse in deviation features

for window in ROLLING_WINDOWS:
    rolling = df["us_aqi"].rolling(window=window, min_periods=window)

    mean_series = rolling.mean()
    rolling_means[window] = mean_series

    new_columns[f"aqi_rolling_mean_{window}"] = mean_series
    new_columns[f"aqi_rolling_min_{window}"] = rolling.min()
    new_columns[f"aqi_rolling_max_{window}"] = rolling.max()
    new_columns[f"aqi_rolling_median_{window}"] = rolling.median()
    new_columns[f"aqi_rolling_std_{window}"] = rolling.std()

print("Rolling statistics created for", len(ROLLING_WINDOWS), "windows.")


# ============================================================
# 15. AQI CHANGE + PERCENTAGE-CHANGE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[9] CREATING AQI CHANGE FEATURES")
print("=" * 70)

for window in CHANGE_WINDOWS:
    new_columns[f"aqi_change_{window}"] = df["us_aqi"].diff(window)

print("Creating AQI percentage-change features...")

for window in PCT_CHANGE_WINDOWS:
    previous_aqi = df["us_aqi"].shift(window)
    new_columns[f"aqi_pct_change_{window}"] = (
        (df["us_aqi"] - previous_aqi) / previous_aqi.replace(0, np.nan)
    ) * 100


# ============================================================
# 16. AQI TREND FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[10] CREATING AQI TREND FEATURES")
print("=" * 70)

for window in TREND_WINDOWS:
    new_columns[f"aqi_trend_{window}"] = df["us_aqi"].diff(window) / window

print("Creating AQI deviation features...")

for window in DEVIATION_WINDOWS:
    new_columns[f"aqi_deviation_from_mean_{window}"] = df["us_aqi"] - rolling_means[window]


# ============================================================
# 17. SAME-HOUR HISTORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[11] CREATING SAME-HOUR HISTORICAL FEATURES")
print("=" * 70)

daily_lags = [df["us_aqi"].shift(24 * days) for days in range(1, 8)]
daily_lag_df = pd.concat(daily_lags, axis=1)

new_columns["aqi_same_hour_mean_7d"] = daily_lag_df.mean(axis=1)
new_columns["aqi_same_hour_std_7d"] = daily_lag_df.std(axis=1)
new_columns["aqi_same_hour_min_7d"] = daily_lag_df.min(axis=1)
new_columns["aqi_same_hour_max_7d"] = daily_lag_df.max(axis=1)


# ============================================================
# 18. POLLUTANT RATIO + COMPOSITION FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[12] CREATING POLLUTANT RATIO FEATURES")
print("=" * 70)

new_columns["pm25_pm10_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)
new_columns["pm25_aqi_ratio"] = df["pm2_5"] / df["us_aqi"].replace(0, np.nan)
new_columns["pm10_aqi_ratio"] = df["pm10"] / df["us_aqi"].replace(0, np.nan)

print("Creating pollutant composition features...")

pm_total = df["pm2_5"] + df["pm10"]
new_columns["pm_total"] = pm_total
new_columns["no2_o3_ratio"] = df["nitrogen_dioxide"] / df["ozone"].replace(0, np.nan)
new_columns["pm25_fraction"] = df["pm2_5"] / pm_total.replace(0, np.nan)


# ============================================================
# 19. WEATHER-POLLUTANT INTERACTIONS
# ============================================================

print("\n" + "=" * 70)
print("[13] CREATING WEATHER-POLLUTANT INTERACTIONS")
print("=" * 70)

new_columns["temperature_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]
new_columns["wind_pm25_interaction"] = df["wind_speed_10m"] * df["pm2_5"]
new_columns["humidity_pm25_interaction"] = df["relative_humidity_2m"] * df["pm2_5"]
new_columns["pressure_pm25_interaction"] = df["surface_pressure"] * df["pm2_5"]
new_columns["temperature_pm25_interaction"] = df["temperature_2m"] * df["pm2_5"]
new_columns["wind_pm10_interaction"] = df["wind_speed_10m"] * df["pm10"]


# ============================================================
# 20. WEATHER / POLLUTANT CHANGE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[14] CREATING WEATHER/POLLUTANT TREND FEATURES")
print("=" * 70)

change_columns = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "cloud_cover",
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
]

for column in change_columns:
    for window in [3, 6, 12, 24]:
        new_columns[f"{column}_change_{window}"] = df[column] - df[column].shift(window)

print(f"Created {len(change_columns) * 4} weather/pollutant change features.")


# ============================================================
# 20B. NEW (V4): FORECAST-LEAD WEATHER FEATURES
# ============================================================
# These are the ACTUAL future weather values at t+24h, t+48h, t+72h,
# standing in for what a live weather-forecast API would report at
# inference time. See the module docstring for why this is legitimate and
# NOT the same kind of leakage as using future AQI.
#
# forecast_<column>_<hours>  =  value of <column> at time (t + hours)

print("\n" + "=" * 70)
print("[14B] CREATING FORECAST-LEAD WEATHER FEATURES (NEW IN V4)")
print("=" * 70)

for column in FORECAST_LEAD_COLUMNS:
    for hours in FORECAST_LEAD_HOURS:
        new_columns[f"forecast_{column}_{hours}"] = df[column].shift(-hours)

print(
    f"Created {len(FORECAST_LEAD_COLUMNS) * len(FORECAST_LEAD_HOURS)} "
    f"forecast-lead features "
    f"({len(FORECAST_LEAD_COLUMNS)} columns x {len(FORECAST_LEAD_HOURS)} horizons)."
)

# Also add the forecasted change in wind/pressure over each horizon, since
# a big pressure DROP or wind speed INCREASE over the forecast window is
# often more predictive of AQI improvement than the raw future value alone.
print("Creating forecast-lead change features (delta between now and forecasted horizon)...")

for hours in FORECAST_LEAD_HOURS:
    new_columns[f"forecast_wind_speed_change_{hours}"] = (
        df["wind_speed_10m"].shift(-hours) - df["wind_speed_10m"]
    )
    new_columns[f"forecast_pressure_change_{hours}"] = (
        df["surface_pressure"].shift(-hours) - df["surface_pressure"]
    )
    new_columns[f"forecast_precipitation_total_{hours}"] = (
        df["precipitation"].rolling(window=hours, min_periods=hours).sum().shift(-hours)
    )

print(f"Created {3 * len(FORECAST_LEAD_HOURS)} forecast-lead change/cumulative features.")


# ============================================================
# 21. WIND DIRECTION COMPONENTS
# ============================================================

print("\nCreating wind direction components...")

new_columns["wind_direction_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360)
new_columns["wind_direction_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360)

print("Creating forecast-lead wind direction components...")

for hours in FORECAST_LEAD_HOURS:
    future_wind_direction = df["wind_direction_10m"].shift(-hours)
    new_columns[f"forecast_wind_direction_sin_{hours}"] = np.sin(2 * np.pi * future_wind_direction / 360)
    new_columns[f"forecast_wind_direction_cos_{hours}"] = np.cos(2 * np.pi * future_wind_direction / 360)


# ============================================================
# 22. ATTACH ALL ENGINEERED FEATURES IN ONE CONCAT
# ============================================================

print("\n" + "=" * 70)
print("[15] ATTACHING ENGINEERED FEATURES (single concat — no fragmentation)")
print("=" * 70)

df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
df = df.copy()   # de-fragment / consolidate memory layout once, cheaply

print("Total engineered feature columns added:", len(new_columns))
print("DataFrame shape after attaching features:", df.shape)


# ============================================================
# 23. FUTURE TARGETS
# ============================================================

print("\n" + "=" * 70)
print("[16] CREATING FUTURE TARGETS")
print("=" * 70)

TARGET_COLUMNS = []

for hours in TARGET_HOURS:
    target_aqi_column = f"target_aqi_{hours}"
    df[target_aqi_column] = df["us_aqi"].shift(-hours)
    TARGET_COLUMNS.append(target_aqi_column)

    target_change_column = f"target_change_{hours}"
    df[target_change_column] = df[target_aqi_column] - df["us_aqi"]
    TARGET_COLUMNS.append(target_change_column)

print("\nTarget columns (all trained on — see train_model.py):")
for column in TARGET_COLUMNS:
    print(" -", column)


# ============================================================
# 24. TARGET ALIGNMENT VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("[17] VERIFYING TARGET ALIGNMENT")
print("=" * 70)

aqi_lookup = df.set_index("time")["us_aqi"]
alignment_failed = False

for hours in TARGET_HOURS:
    target_column = f"target_aqi_{hours}"
    expected = (df["time"] + pd.Timedelta(hours=hours)).map(aqi_lookup)
    valid = expected.notna()
    comparison = df[target_column] == expected

    correct = int(comparison[valid].sum())
    incorrect = int((~comparison[valid]).sum())

    print(f"\n{hours}-HOUR TARGET")
    print("-" * 40)
    print("Rows checked:", int(valid.sum()))
    print("Correct:", correct)
    print("Incorrect:", incorrect)

    if incorrect > 0:
        alignment_failed = True

if alignment_failed:
    raise ValueError("\nTARGET ALIGNMENT FAILED.\nDo not train the model.")

print("\nAll AQI target alignment checks: PASS")


# ============================================================
# 24B. NEW (V4): FORECAST-LEAD FEATURE ALIGNMENT VERIFICATION
# ============================================================
# Same idea as target alignment above, but for the new forecast_* columns:
# forecast_<col>_<h> at row time t must equal the raw <col> value at time
# t + h. This guarantees the shift(-h) was applied correctly and that no
# accidental off-by-one or timezone issue crept in.

print("\n" + "=" * 70)
print("[17B] VERIFYING FORECAST-LEAD FEATURE ALIGNMENT")
print("=" * 70)

forecast_alignment_failed = False

for column in FORECAST_LEAD_COLUMNS:
    column_lookup = df.set_index("time")[column]
    for hours in FORECAST_LEAD_HOURS:
        feature_name = f"forecast_{column}_{hours}"
        expected = (df["time"] + pd.Timedelta(hours=hours)).map(column_lookup)
        valid = expected.notna() & df[feature_name].notna()

        if valid.sum() == 0:
            continue

        comparison = np.isclose(
            df.loc[valid, feature_name].to_numpy(),
            expected.loc[valid].to_numpy(),
            equal_nan=True,
        )
        incorrect = int((~comparison).sum())

        if incorrect > 0:
            print(f"FAIL: {feature_name} — {incorrect} misaligned rows out of {int(valid.sum())} checked")
            forecast_alignment_failed = True

if forecast_alignment_failed:
    raise ValueError("\nFORECAST-LEAD FEATURE ALIGNMENT FAILED.\nDo not train the model.")

print(f"All {len(FORECAST_LEAD_COLUMNS) * len(FORECAST_LEAD_HOURS)} forecast-lead alignment checks: PASS")


# ============================================================
# 25. TARGET CHANGE VERIFICATION
# ============================================================

print("\nChecking target-change alignment...")

for hours in TARGET_HOURS:
    target_aqi_column = f"target_aqi_{hours}"
    target_change_column = f"target_change_{hours}"

    expected_change = df[target_aqi_column] - df["us_aqi"]
    valid = expected_change.notna()

    comparison = np.isclose(
        df.loc[valid, target_change_column],
        expected_change.loc[valid],
    )
    incorrect = int((~comparison).sum())

    if incorrect > 0:
        raise ValueError(f"{target_change_column} alignment failed.")

    print(f"{target_change_column}: PASS")


# ============================================================
# 26. CHECK MISSING VALUES (pre-cleaning report)
# ============================================================

print("\n" + "=" * 70)
print("[18] CHECKING MISSING VALUES")
print("=" * 70)

missing_counts = df.isnull().sum()
missing_report = missing_counts[missing_counts > 0].sort_values(ascending=False)

if len(missing_report) > 0:
    print("\nMissing values before cleaning (top 15):")
    print(missing_report.head(15))
else:
    print("No missing values.")


# ============================================================
# 27. CHECK INFINITE VALUES
# ============================================================

print("\nChecking infinite values...")

numeric_columns = df.select_dtypes(include=np.number).columns
infinite_count = int(np.isinf(df[numeric_columns]).sum().sum())

print("Infinite values:", infinite_count)

if infinite_count > 0:
    df = df.replace([np.inf, -np.inf], np.nan)
    print("Infinite values converted to NaN.")


# ============================================================
# 28. DROP NaN ROWS
# ============================================================

print("\n" + "=" * 70)
print("[19] CLEANING DATASET")
print("=" * 70)

before_drop = len(df)
print("\n" + "=" * 70)
print("NaN ROW ANALYSIS BEFORE DROPNA")
print("=" * 70)

nan_rows = df.isna().any(axis=1)

print("Total rows:", len(df))
print("Rows with NaN:", nan_rows.sum())
print("Rows without NaN:", (~nan_rows).sum())

print("\nNaN count by column:")
print(
    df.isna()
      .sum()
      .sort_values(ascending=False)
      .head(30)
)

print("\nFirst rows containing NaN:")
print(df.loc[nan_rows, ["time"]].head(10))

print("\nLast rows containing NaN:")
print(df.loc[nan_rows, ["time"]].tail(10))
df = df.dropna().reset_index(drop=True)
after_drop = len(df)

print("Rows before :", before_drop)
print("Rows after  :", after_drop)
print("Rows removed:", before_drop - after_drop)
print("\nMaximum historical lag:", max(LAG_HOURS), "hours")
print("Maximum future target:", max(TARGET_HOURS), "hours")
print("Maximum forecast-lead horizon:", max(FORECAST_LEAD_HOURS), "hours")
print(
    "\nNote: forecast-lead features use shift(-h) with the same max horizon "
    "(72h) as the targets, so they do not remove any additional rows beyond "
    "what the targets already require."
)

if after_drop < 500:
    print(
        "\nWARNING: Fewer than 500 rows remain after dropping NaNs. "
        "Consider backfilling more history."
    )


# ============================================================
# 29. FINAL TARGET VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("[20] FINAL TARGET VALIDATION")
print("=" * 70)

for target in TARGET_COLUMNS:
    target_valid = df[target].notna().all()
    print(f"{target}:", "PASS" if target_valid else "FAIL")
    if not target_valid:
        raise ValueError(f"{target} contains NaN values.")


# ============================================================
# 30. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("[21] FINAL DATA QUALITY CHECK")
print("=" * 70)

final_missing = int(df.isnull().sum().sum())
final_duplicates = int(df["time"].duplicated().sum())
final_infinite = int(np.isinf(df.select_dtypes(include=np.number)).sum().sum())

print("Missing values       :", final_missing)
print("Duplicate timestamps :", final_duplicates)
print("Infinite values      :", final_infinite)
print("Rows                 :", len(df))
print("Columns              :", len(df.columns))
print("Date range           :", df["time"].min(), "→", df["time"].max())

if final_missing != 0:
    raise ValueError("Final dataset still contains missing values.")
if final_duplicates != 0:
    raise ValueError("Final dataset contains duplicate timestamps.")
if final_infinite != 0:
    raise ValueError("Final dataset contains infinite values.")


# ============================================================
# 31. IDENTIFY MODEL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("[22] IDENTIFYING MODEL FEATURES")
print("=" * 70)

# IMPORTANT: these columns contain future information about AQI itself (the
# label) and MUST NOT be given to the model as input. Note that forecast_*
# columns are intentionally NOT excluded here — they carry future WEATHER
# information (not AQI), which is legitimately knowable in advance via a
# forecast API, and are the whole point of this V4 update.
EXCLUDED_FROM_FEATURES = ["time"] + TARGET_COLUMNS

FEATURE_COLUMNS = [c for c in df.columns if c not in EXCLUDED_FROM_FEATURES]

forecast_lead_feature_count = len([c for c in FEATURE_COLUMNS if c.startswith("forecast_")])

print("Total columns   :", len(df.columns))
print("Feature columns :", len(FEATURE_COLUMNS))
print("  (of which forecast-lead weather features:", forecast_lead_feature_count, ")")
print("Target columns  :", len(TARGET_COLUMNS))
print("\nExcluded columns:")
for c in EXCLUDED_FROM_FEATURES:
    print(" -", c)


# ============================================================
# 32. SAFETY CHECK FOR TARGET LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("[23] TARGET LEAKAGE CHECK")
print("=" * 70)

leakage_columns = [c for c in TARGET_COLUMNS if c in FEATURE_COLUMNS]

if len(leakage_columns) > 0:
    raise ValueError(
        f"\nTARGET LEAKAGE DETECTED!\nThese targets are present in features:\n{leakage_columns}"
    )

print("Target leakage check: PASS")
print("No future target columns are present in FEATURE_COLUMNS.")

# NEW (V4): explicitly confirm forecast_* columns never touch us_aqi /
# european_aqi — they should be built purely from weather columns, never
# from AQI, since AQI itself is not available from a forecast API.
suspicious_forecast_columns = [
    c for c in FEATURE_COLUMNS
    if c.startswith("forecast_") and ("aqi" in c.lower())
]
if suspicious_forecast_columns:
    raise ValueError(
        f"\nFORECAST-LEAD FEATURE LEAKAGE DETECTED!\n"
        f"These forecast_* columns appear to be derived from AQI, not "
        f"weather, which would leak the label:\n{suspicious_forecast_columns}"
    )
print("Forecast-lead AQI-leakage check: PASS (no forecast_* column derived from AQI).")


# ============================================================
# 33. SAVE FEATURE DATASET + FEATURE LIST
# ============================================================

print("\n" + "=" * 70)
print("[24] SAVING FEATURE DATASET")
print("=" * 70)

os.makedirs(PROCESSED_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print("Feature dataset saved to:", OUTPUT_FILE)

with open(FEATURE_LIST_FILE, "w") as f:
    json.dump(FEATURE_COLUMNS, f, indent=4)
print("Feature list saved to:", FEATURE_LIST_FILE)


# ============================================================
# 34. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING V4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal shape:", df.shape)
print("Feature columns:", len(FEATURE_COLUMNS))
print("  (forecast-lead weather features:", forecast_lead_feature_count, ")")
print("Target columns :", len(TARGET_COLUMNS), "(all 3 horizons — 24h/48h/72h — trained on)")
print("Missing values :", df.isnull().sum().sum())
print("Date range     :", df["time"].min(), "→", df["time"].max())

print("\nFirst 5 rows:")
sample_columns = [
    "time", "us_aqi", "aqi_lag_24", "aqi_lag_48", "aqi_lag_72",
    "aqi_rolling_mean_24", "aqi_trend_24",
    "forecast_wind_speed_10m_24", "forecast_precipitation_total_24",
    "target_aqi_24", "target_aqi_48", "target_aqi_72",
]
print(df[sample_columns].head())


# ============================================================
# 35. NEXT STEPS
# ============================================================

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print(
    """
1. Run validate_features.py to verify the V4 dataset (note: it will need a
   small update to also validate the new forecast_* columns — ask if you
   want that script updated too).
2. Upload karachi_features_v3.csv to Hopsworks as a NEW Feature Group
   version (bump FEATURE_GROUP_VERSION in feature_store.py, e.g. 6 -> 7,
   since the schema changed: new forecast_* columns were added).
3. Recreate the Feature View for the new version in feature_view.py (bump
   FEATURE_GROUP_VERSION and FEATURE_VIEW_VERSION to match) — still
   excluding target_change_* from the query.
4. Update train_model.py's FEATURE_GROUP_VERSION to match, then re-run —
   it trains SEPARATE models per horizon for all three targets.
5. IMPORTANT: whatever builds real-time predictions (predict.py / the
   dashboard) must populate forecast_* columns from a LIVE call to
   fetch_current_window() / the Open-Meteo forecast API — NOT from
   historical data, which won't exist yet for future timestamps.
6. Compare Random Forest / Ridge / XGBoost / CatBoost / Neural Network
   using RMSE, MAE and R² per horizon — expect the biggest gains at 48h
   and especially 72h, since those horizons benefited least from the
   change-target fix and most from finally knowing future weather.
"""
)
print("=" * 70)
print("READY FOR HOPSWORKS FEATURE STORE")
print("=" * 70)