"""
live_features.py

Builds ONE feature row representing "right now" for real-time AQI
prediction — the missing piece between the training pipeline and
predict.py.

============================================================
WHY THIS FILE EXISTS
============================================================
The training Feature Group (via feature_engineering.py) drops every row
that doesn't have a valid target_aqi_72 — i.e. a real AQI reading 72
hours in the future. That's correct for TRAINING: you can't train on a
target you don't have. But it means the newest surviving row in the
Feature Group / Feature View is ALWAYS ~72 hours behind the current
moment. If predict.py reads "the latest row" from there, it is
reporting stale AQI as "current" on every single run — not a one-off
bug, a structural guarantee of that data path.

This module builds today's row a different way:
    1. Load the FULL local processed history (karachi_processed.csv) —
       gives us real actual AQI/weather back to 2023, enough for even
       the 504h (21-day) lag feature.
    2. Fetch a LIVE current window from Open-Meteo (fetch_current_window
       from fetch_data.py) — gives us today's actual-ish conditions PLUS
       several days of genuine forecast for temperature/wind/pressure/etc.
    3. Merge them (live data wins on overlapping hours — it's fresher).
    4. Run the EXACT SAME feature formulas as feature_engineering.py
       (lags, rolling stats, changes, trends, same-hour history,
       pollutant ratios, weather interactions, forecast-lead features)
       across the merged series.
    5. Extract just the single row at "now" and return it, already
       restricted + ordered to match a given feature_columns.json.

Because step 4 reuses identical formulas to training, there is no risk
of "training features computed one way, live features computed a
slightly different way" drift — the single biggest source of silent
train/serve skew bugs.

============================================================
USAGE (from predict.py)
============================================================
    from live_features import build_live_feature_row

    row_df, current_aqi, now_timestamp = build_live_feature_row(
        feature_columns_path=r"saved_models/fs70/feature_columns.json",
    )

    # row_df is a single-row DataFrame, columns already matching the
    # order in feature_columns.json — feed directly into your models:
    predicted_change_24 = catboost_models["target_aqi_24"].predict(row_df)

    # Reconstruct absolute AQI the same way train_model.py does:
    predicted_aqi_24 = current_aqi + predicted_change_24
"""

import os
import sys
import json
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================
# This file is expected to sit at the PROJECT ROOT (same level as
# predict.py), with the feature pipeline in a "feature_pipeline"
# subfolder — matching the layout implied by your other scripts.
# Adjust FEATURE_PIPELINE_DIRNAME below if yours differs.

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR

FEATURE_PIPELINE_DIRNAME = "feature_pipeline"
FEATURE_PIPELINE_DIR = os.path.join(PROJECT_ROOT, FEATURE_PIPELINE_DIRNAME)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw_dataset")
PROCESSED_HISTORY_FILE = os.path.join(RAW_DIR, "karachi_processed.csv")

# Karachi has no DST, fixed UTC+5 year-round, so this is safe to hardcode.
LOCAL_TZ = "Asia/Karachi"


# ============================================================
# IMPORT fetch_current_window() FROM THE FEATURE PIPELINE
# ============================================================

if FEATURE_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, FEATURE_PIPELINE_DIR)

from fetch_data import fetch_current_window  # noqa: E402  (path inserted above)


# ============================================================
# CONFIG — MUST STAY IN SYNC WITH feature_engineering.py
# ============================================================
# If you ever change these lists in feature_engineering.py, copy the
# change here too, or live features will silently stop matching what
# the models were trained on.

LAG_HOURS = [1, 3, 6, 12, 24, 48, 72, 96, 120, 168, 336, 504]
ROLLING_WINDOWS = [6, 12, 24, 48, 72, 168]
CHANGE_WINDOWS = [1, 3, 6, 12, 24, 48, 72]
PCT_CHANGE_WINDOWS = [3, 6, 12, 24, 48, 72, 168]
TREND_WINDOWS = [6, 12, 24, 48, 72, 168]
DEVIATION_WINDOWS = [24, 48, 72, 168]

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

WEATHER_POLLUTANT_CHANGE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "cloud_cover",
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
]
WEATHER_POLLUTANT_CHANGE_WINDOWS = [3, 6, 12, 24]

REQUIRED_COLUMNS = [
    "time", "us_aqi",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "cloud_cover", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "weather_code",
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "european_aqi",
]


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# STEP 1: LOAD LOCAL HISTORY
# ============================================================

def _load_local_history():
    if not os.path.exists(PROCESSED_HISTORY_FILE):
        raise FileNotFoundError(
            f"Local processed history not found:\n{PROCESSED_HISTORY_FILE}\n"
            "Run fetch_data.py -> preprocess.py -> trim_future_rows.py first."
        )

    hist = pd.read_csv(PROCESSED_HISTORY_FILE)
    hist["time"] = pd.to_datetime(hist["time"], format="mixed", errors="coerce")
    hist = hist.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return hist


# ============================================================
# STEP 2: FETCH LIVE WINDOW (actual "now" + genuine forecast ahead)
# ============================================================

def _fetch_live_window():
    live = fetch_current_window()
    if live is None or live.empty:
        raise RuntimeError(
            "fetch_current_window() returned no data — check your internet "
            "connection / Open-Meteo API availability."
        )
    return live


# ============================================================
# STEP 3: MERGE (live wins on overlap — it's the freshest reading)
# ============================================================

def _merge_history_and_live(hist, live):
    combined = pd.concat([hist, live], ignore_index=True)

    combined = combined[[c for c in combined.columns if c in REQUIRED_COLUMNS]]

    combined = combined.drop_duplicates(subset=["time"], keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)

    # Reindex onto a complete hourly range, same fix as feature_engineering.py,
    # so shift()-based lag/rolling math lines up correctly even if a hour is
    # missing somewhere in the middle of local history.
    full_range = pd.date_range(
        start=combined["time"].min(),
        end=combined["time"].max(),
        freq="h",
    )
    combined = (
        combined.set_index("time")
        .reindex(full_range)
        .rename_axis("time")
        .reset_index()
    )

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "time"]
    for c in numeric_cols:
        combined[c] = pd.to_numeric(combined[c], errors="coerce")

    return combined


# ============================================================
# STEP 4: COMPUTE FEATURES (mirrors feature_engineering.py exactly)
# ============================================================

def _compute_all_features(df):
    new_columns = {}

    # ---------------- TIME FEATURES ----------------
    hour = df["time"].dt.hour
    day = df["time"].dt.day
    month = df["time"].dt.month
    day_of_week = df["time"].dt.dayofweek
    day_of_year = df["time"].dt.dayofyear
    week_of_year = df["time"].dt.isocalendar().week.astype(int)
    is_weekend = (day_of_week >= 5).astype(int)

    new_columns.update({
        "hour": hour, "day": day, "month": month,
        "day_of_week": day_of_week, "day_of_year": day_of_year,
        "week_of_year": week_of_year, "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7),
        "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
    })

    # ---------------- AQI LAGS ----------------
    for lag in LAG_HOURS:
        new_columns[f"aqi_lag_{lag}"] = df["us_aqi"].shift(lag)

    # ---------------- ROLLING AQI STATS ----------------
    rolling_means = {}
    for window in ROLLING_WINDOWS:
        rolling = df["us_aqi"].rolling(window=window, min_periods=window)
        mean_series = rolling.mean()
        rolling_means[window] = mean_series
        new_columns[f"aqi_rolling_mean_{window}"] = mean_series
        new_columns[f"aqi_rolling_min_{window}"] = rolling.min()
        new_columns[f"aqi_rolling_max_{window}"] = rolling.max()
        new_columns[f"aqi_rolling_median_{window}"] = rolling.median()
        new_columns[f"aqi_rolling_std_{window}"] = rolling.std()

    # ---------------- CHANGE / PCT CHANGE ----------------
    for window in CHANGE_WINDOWS:
        new_columns[f"aqi_change_{window}"] = df["us_aqi"].diff(window)

    for window in PCT_CHANGE_WINDOWS:
        previous_aqi = df["us_aqi"].shift(window)
        new_columns[f"aqi_pct_change_{window}"] = (
            (df["us_aqi"] - previous_aqi) / previous_aqi.replace(0, np.nan)
        ) * 100

    # ---------------- TREND / DEVIATION ----------------
    for window in TREND_WINDOWS:
        new_columns[f"aqi_trend_{window}"] = df["us_aqi"].diff(window) / window

    for window in DEVIATION_WINDOWS:
        new_columns[f"aqi_deviation_from_mean_{window}"] = df["us_aqi"] - rolling_means[window]

    # ---------------- SAME-HOUR HISTORICAL ----------------
    daily_lags = [df["us_aqi"].shift(24 * days) for days in range(1, 8)]
    daily_lag_df = pd.concat(daily_lags, axis=1)
    new_columns["aqi_same_hour_mean_7d"] = daily_lag_df.mean(axis=1)
    new_columns["aqi_same_hour_std_7d"] = daily_lag_df.std(axis=1)
    new_columns["aqi_same_hour_min_7d"] = daily_lag_df.min(axis=1)
    new_columns["aqi_same_hour_max_7d"] = daily_lag_df.max(axis=1)

    # ---------------- POLLUTANT RATIOS / COMPOSITION ----------------
    new_columns["pm25_pm10_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)
    new_columns["pm25_aqi_ratio"] = df["pm2_5"] / df["us_aqi"].replace(0, np.nan)
    new_columns["pm10_aqi_ratio"] = df["pm10"] / df["us_aqi"].replace(0, np.nan)

    pm_total = df["pm2_5"] + df["pm10"]
    new_columns["pm_total"] = pm_total
    new_columns["no2_o3_ratio"] = df["nitrogen_dioxide"] / df["ozone"].replace(0, np.nan)
    new_columns["pm25_fraction"] = df["pm2_5"] / pm_total.replace(0, np.nan)

    # ---------------- WEATHER-POLLUTANT INTERACTIONS ----------------
    new_columns["temperature_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]
    new_columns["wind_pm25_interaction"] = df["wind_speed_10m"] * df["pm2_5"]
    new_columns["humidity_pm25_interaction"] = df["relative_humidity_2m"] * df["pm2_5"]
    new_columns["pressure_pm25_interaction"] = df["surface_pressure"] * df["pm2_5"]
    new_columns["temperature_pm25_interaction"] = df["temperature_2m"] * df["pm2_5"]
    new_columns["wind_pm10_interaction"] = df["wind_speed_10m"] * df["pm10"]

    # ---------------- WEATHER / POLLUTANT CHANGE ----------------
    for column in WEATHER_POLLUTANT_CHANGE_COLUMNS:
        for window in WEATHER_POLLUTANT_CHANGE_WINDOWS:
            new_columns[f"{column}_change_{window}"] = df[column] - df[column].shift(window)

    # ---------------- FORECAST-LEAD WEATHER FEATURES ----------------
    for column in FORECAST_LEAD_COLUMNS:
        for hours in FORECAST_LEAD_HOURS:
            new_columns[f"forecast_{column}_{hours}"] = df[column].shift(-hours)

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

    # ---------------- WIND DIRECTION COMPONENTS ----------------
    new_columns["wind_direction_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360)
    new_columns["wind_direction_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360)

    for hours in FORECAST_LEAD_HOURS:
        future_wind_direction = df["wind_direction_10m"].shift(-hours)
        new_columns[f"forecast_wind_direction_sin_{hours}"] = np.sin(2 * np.pi * future_wind_direction / 360)
        new_columns[f"forecast_wind_direction_cos_{hours}"] = np.cos(2 * np.pi * future_wind_direction / 360)

    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df.copy()


# ============================================================
# STEP 5: PUBLIC ENTRY POINT
# ============================================================

def build_live_feature_row(feature_columns_path, verbose=True):
    """
    Returns:
        row_df         : single-row DataFrame, columns matching the
                          order in feature_columns_path, ready to feed
                          straight into a loaded model.
        current_aqi    : float, the actual us_aqi at "now" — use this
                          for predicted_aqi = current_aqi + predicted_change.
        now_timestamp  : pandas.Timestamp used as "now" (naive, local
                          Karachi time, floored to the hour).
    """

    if verbose:
        print_section("BUILDING LIVE FEATURE ROW")

    now = pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).floor("h")
    if verbose:
        print("Reference 'now' (Asia/Karachi, floored to hour):", now)

    hist = _load_local_history()
    if verbose:
        print("Local history rows:", len(hist), "| up to:", hist["time"].max())

    live = _fetch_live_window()
    if verbose:
        print("Live window rows  :", len(live), "| range:", live["time"].min(), "->", live["time"].max())

    combined = _merge_history_and_live(hist, live)
    if verbose:
        print("Combined rows     :", len(combined), "| range:", combined["time"].min(), "->", combined["time"].max())

    if now > combined["time"].max():
        # Live fetch didn't include the current hour yet (API lag) — fall
        # back to the latest hour actually available instead of failing.
        now = combined["time"].max()
        if verbose:
            print("WARNING: live data doesn't reach current wall-clock hour yet.")
            print("Falling back to latest available hour as 'now':", now)

    featured = _compute_all_features(combined)

    now_row = featured[featured["time"] == now]
    if now_row.empty:
        raise ValueError(
            f"No row found for timestamp {now} after feature computation. "
            "Check for gaps in the merged history/live data."
        )
    now_row = now_row.iloc[[0]]

    current_aqi = float(now_row["us_aqi"].iloc[0])

    with open(feature_columns_path, "r") as f:
        feature_columns = json.load(f)

    missing = [c for c in feature_columns if c not in now_row.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} expected feature columns are missing from the "
            f"live row (check LAG_HOURS/ROLLING_WINDOWS/etc. above are in "
            f"sync with feature_engineering.py): {missing[:10]}"
        )

    row_df = now_row[feature_columns].reset_index(drop=True)

    nan_cols = row_df.columns[row_df.isna().any()].tolist()
    if nan_cols:
        raise ValueError(
            f"{len(nan_cols)} required features are NaN for 'now' — usually "
            f"means not enough local history for a long lag/rolling window, "
            f"or the live forecast doesn't reach far enough ahead. "
            f"Affected columns: {nan_cols[:10]}"
        )

    if verbose:
        print("\nCurrent us_aqi:", current_aqi)
        print("Live feature row ready:", row_df.shape)

    return row_df, current_aqi, now


if __name__ == "__main__":
    # Quick manual test — point this at whichever feature set you're serving.
    test_feature_columns_path = os.path.join(
        PROJECT_ROOT, "saved_models", "fs70", "feature_columns.json"
    )
    row_df, current_aqi, now_timestamp = build_live_feature_row(test_feature_columns_path)
    print("\n" + "=" * 60)
    print("TEST RESULT")
    print("=" * 60)
    print("Now:", now_timestamp)
    print("Current AQI:", current_aqi)
    print("Row shape:", row_df.shape)
    print(row_df.head())