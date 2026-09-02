"""
fetch_data.py

Karachi Air Quality + Weather Data Fetcher

This script:

1. Fetches historical weather + air quality data when manually enabled.
2. Fetches the current rolling forecast window.
3. Appends new data to the existing raw dataset.
4. Removes duplicate timestamps.
5. Sorts data chronologically.
6. Handles different timestamp formats safely.
7. Uses an absolute project-root-based file path so it works
   regardless of the current terminal directory.

Dataset:
    data/raw_dataset/karachi_raw.csv

IMPORTANT:
    - Historical backfill should normally be run ONCE.
    - After the initial backfill, keep the backfill section commented.
    - The normal script fetches the current rolling window.

NOTE (this version):
    The historical backfill section below is UNCOMMENTED and set to fetch
    3 years of data (2023-08-28 -> 2026-08-28). Run this ONCE to build your
    initial raw dataset, then RE-COMMENT the backfill call again before you
    go back to normal hourly/current-window runs -- otherwise every run
    will re-fetch 3 years of history unnecessarily (slow + wasteful, though
    save_raw_dataset()'s dedup logic will still keep the file correct).
"""


import os
import requests
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# Location of this file:
# Air_Quality_Index_Predictor/
# └── Feature Pipeline/
#     └── fetch_data.py

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Move one level up:
# Feature Pipeline -> Air_Quality_Index_Predictor

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

# Final data directory
RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw_dataset"
)

# Final CSV path
RAW_FILE = os.path.join(
    RAW_DIR,
    "karachi_raw.csv"
)


# ============================================================
# LOCATION
# ============================================================

LATITUDE = 24.8607
LONGITUDE = 67.0011


# ============================================================
# WEATHER VARIABLES
# ============================================================

WEATHER_HOURLY_VARS = (
    "temperature_2m,"
    "wind_direction_10m,"
    "wind_speed_10m,"
    "cloud_cover,"
    "precipitation,"
    "relative_humidity_2m,"
    "surface_pressure,"
    "weather_code,"
    "visibility,"
    "dew_point_2m"
)


# ============================================================
# AIR QUALITY VARIABLES
# ============================================================

AQ_HOURLY_VARS = (
    "pm10,"
    "pm2_5,"
    "european_aqi,"
    "carbon_monoxide,"
    "nitrogen_dioxide,"
    "sulphur_dioxide,"
    "ozone,"
    "us_aqi"
)


# ============================================================
# FORECAST WEATHER API
# ============================================================

FORECAST_WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast?"
    f"latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&hourly={WEATHER_HOURLY_VARS}"
    "&timezone=auto"
    "&past_days=3"
)


# ============================================================
# FORECAST AIR QUALITY API
# ============================================================

FORECAST_AQ_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&hourly={AQ_HOURLY_VARS}"
    "&timezone=auto"
    "&past_days=3" 
)


# ============================================================
# GENERIC API FETCH FUNCTION
# ============================================================

def fetch_json(url, label):
    """
    Fetch JSON data from an API.

    Returns:
        dict if successful
        None if request fails
    """

    try:

        response = requests.get(
            url,
            timeout=30
        )

    except requests.exceptions.RequestException as e:

        print(
            f"ERROR: Request failed for {label}: {e}"
        )

        return None

    # --------------------------------------------------------
    # Successful response
    # --------------------------------------------------------

    if response.status_code == 200:

        print(
            f"{label} fetched successfully"
        )

        return response.json()

    # --------------------------------------------------------
    # Failed response
    # --------------------------------------------------------

    print(
        f"ERROR: {label} fetch failed "
        f"with status {response.status_code}"
    )

    print(
        response.text[:500]
    )

    return None


# ============================================================
# CURRENT / ROLLING WINDOW
# ============================================================

def fetch_current_window():
    """
    Fetch current rolling weather and air quality data.

    Returns:
        Merged pandas DataFrame
        None if fetching fails
    """

    # --------------------------------------------------------
    # Fetch weather
    # --------------------------------------------------------

    weather_data = fetch_json(
        FORECAST_WEATHER_URL,
        "Weather data"
    )

    # --------------------------------------------------------
    # Fetch air quality
    # --------------------------------------------------------

    air_quality_data = fetch_json(
        FORECAST_AQ_URL,
        "Air quality data"
    )

    # --------------------------------------------------------
    # Check API responses
    # --------------------------------------------------------

    if weather_data is None:

        print(
            "ERROR: Weather data unavailable."
        )

        return None

    if air_quality_data is None:

        print(
            "ERROR: Air quality data unavailable."
        )

        return None

    # --------------------------------------------------------
    # Check hourly key
    # --------------------------------------------------------

    if "hourly" not in weather_data:

        print(
            "ERROR: Weather API response "
            "does not contain 'hourly'."
        )

        return None

    if "hourly" not in air_quality_data:

        print(
            "ERROR: Air quality API response "
            "does not contain 'hourly'."
        )

        return None

    # --------------------------------------------------------
    # Convert API response to DataFrames
    # --------------------------------------------------------

    weather_df = pd.DataFrame(
        weather_data["hourly"]
    )

    air_quality_df = pd.DataFrame(
        air_quality_data["hourly"]
    )

    # --------------------------------------------------------
    # Merge weather + air quality
    # --------------------------------------------------------

    merged_df = pd.merge(
        weather_df,
        air_quality_df,
        how="inner",
        on="time"
    )

    # --------------------------------------------------------
    # Convert timestamp safely
    # --------------------------------------------------------

    merged_df["time"] = pd.to_datetime(
        merged_df["time"],
        format="mixed",
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------------

    merged_df = merged_df.dropna(
        subset=["time"]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    merged_df = (
        merged_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    print(
        "Current window rows:",
        len(merged_df)
    )

    if len(merged_df) > 0:

        print(
            "Current window date range:",
            merged_df["time"].min(),
            "→",
            merged_df["time"].max()
        )

    return merged_df


# ============================================================
# HISTORICAL BACKFILL
# ============================================================

def fetch_historical_data(
    start_date,
    end_date
):
    """
    Fetch historical weather + air quality data.

    Example:

        fetch_historical_data(
            "2025-06-01",
            "2026-08-25"
        )

    Dates must use:

        YYYY-MM-DD
    """

    # --------------------------------------------------------
    # Historical weather URL
    # --------------------------------------------------------

    weather_url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&hourly={WEATHER_HOURLY_VARS}"
        "&timezone=auto"
    )

    # --------------------------------------------------------
    # Historical air quality URL
    # --------------------------------------------------------

    aq_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&hourly={AQ_HOURLY_VARS}"
        "&timezone=auto"
    )

    # --------------------------------------------------------
    # Fetch weather
    # --------------------------------------------------------

    weather_data = fetch_json(
        weather_url,
        "Historical weather data"
    )

    # --------------------------------------------------------
    # Fetch AQ
    # --------------------------------------------------------

    air_quality_data = fetch_json(
        aq_url,
        "Historical air quality data"
    )

    # --------------------------------------------------------
    # Check responses
    # --------------------------------------------------------

    if weather_data is None:

        print(
            "ERROR: Historical weather fetch failed."
        )

        return None

    if air_quality_data is None:

        print(
            "ERROR: Historical air quality fetch failed."
        )

        return None

    # --------------------------------------------------------
    # Check hourly data
    # --------------------------------------------------------

    if "hourly" not in weather_data:

        print(
            "ERROR: Historical weather response "
            "missing 'hourly'."
        )

        return None

    if "hourly" not in air_quality_data:

        print(
            "ERROR: Historical AQ response "
            "missing 'hourly'."
        )

        return None

    # --------------------------------------------------------
    # Convert to DataFrames
    # --------------------------------------------------------

    weather_df = pd.DataFrame(
        weather_data["hourly"]
    )

    air_quality_df = pd.DataFrame(
        air_quality_data["hourly"]
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged_df = pd.merge(
        weather_df,
        air_quality_df,
        how="inner",
        on="time"
    )

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    merged_df["time"] = pd.to_datetime(
        merged_df["time"],
        format="mixed",
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------------

    merged_df = merged_df.dropna(
        subset=["time"]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    merged_df = (
        merged_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    print(
        "Historical rows fetched:",
        len(merged_df)
    )

    if len(merged_df) > 0:

        print(
            "Historical date range:",
            merged_df["time"].min(),
            "→",
            merged_df["time"].max()
        )

    return merged_df


# ============================================================
# SAVE RAW DATASET
# ============================================================

def save_raw_dataset(new_df):
    """
    Append new data to existing dataset.

    Operations:

        Existing CSV
              +
        New data
              ↓
        Combine
              ↓
        Convert timestamps
              ↓
        Remove invalid timestamps
              ↓
        Remove duplicate timestamps
              ↓
        Sort chronologically
              ↓
        Save CSV
    """

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if new_df is None:

        print(
            "ERROR: No data received."
        )

        return

    if new_df.empty:

        print(
            "ERROR: DataFrame is empty."
        )

        return

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    os.makedirs(
        RAW_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load existing dataset
    # --------------------------------------------------------

    if os.path.exists(RAW_FILE):

        print(
            "\nExisting dataset found:"
        )

        print(
            RAW_FILE
        )

        existing_df = pd.read_csv(
            RAW_FILE
        )

        print(
            "Existing rows:",
            len(existing_df)
        )

        # Combine old + new
        combined_df = pd.concat(
            [
                existing_df,
                new_df
            ],
            ignore_index=True
        )

    else:

        print(
            "\nNo existing dataset found."
        )

        print(
            "Creating a new dataset."
        )

        combined_df = new_df.copy()

    # --------------------------------------------------------
    # Convert timestamps safely
    #
    # Handles:
    #
    # 2026-08-26T00:00
    #
    # and:
    #
    # 2026-08-25 23:00:00
    # --------------------------------------------------------

    combined_df["time"] = pd.to_datetime(
        combined_df["time"],
        format="mixed",
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------------

    invalid_count = (
        combined_df["time"]
        .isna()
        .sum()
    )

    if invalid_count > 0:

        print(
            "WARNING:",
            invalid_count,
            "invalid timestamps removed."
        )

        combined_df = combined_df.dropna(
            subset=["time"]
        )

    # --------------------------------------------------------
    # Count duplicates
    # --------------------------------------------------------

    rows_before_dedup = len(
        combined_df
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    combined_df = combined_df.drop_duplicates(
        subset=["time"],
        keep="last"
    )

    rows_after_dedup = len(
        combined_df
    )

    duplicates_removed = (
        rows_before_dedup
        - rows_after_dedup
    )

    print(
        "Duplicate rows removed:",
        duplicates_removed
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    combined_df = (
        combined_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save timestamp in standard format
    # --------------------------------------------------------

    combined_df["time"] = (
        combined_df["time"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    combined_df.to_csv(
        RAW_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RAW DATASET SAVED SUCCESSFULLY")
    print("=" * 60)

    print(
        "Total rows:",
        len(combined_df)
    )

    print(
        "Total columns:",
        len(combined_df.columns)
    )

    print(
        "Date range:",
        combined_df["time"].iloc[0],
        "→",
        combined_df["time"].iloc[-1]
    )

    print(
        "File:",
        RAW_FILE
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("KARACHI AQI DATA FETCHER")
    print("=" * 60)

    print(
        "\nProject root:",
        PROJECT_ROOT
    )

    print(
        "Raw dataset:",
        RAW_FILE
    )

    # ========================================================
    # INITIAL HISTORICAL BACKFILL (3 YEARS)
    # ========================================================
    #
    # IMPORTANT:
    #
    # This is UNCOMMENTED right now to pull 3 years of history
    # (2023-08-28 -> 2026-08-28). Run the script ONCE like this.
    #
    # Open-Meteo's archive API can be picky about very large single
    # requests -- if this call errors out or times out, split it into
    # 1-year chunks instead (call fetch_historical_data() three times
    # with 2023-08-28->2024-08-28, 2024-08-28->2025-08-28,
    # 2025-08-28->2026-08-28, calling save_raw_dataset() after each
    # chunk) and re-run this file three times, or wrap the three calls
    # in a loop.
    #
    # After this backfill run completes successfully, RE-COMMENT this
    # block again before going back to normal hourly/current-window
    # runs -- otherwise every future run will re-fetch 3 years of
    # history unnecessarily.
    #
    # ========================================================

    # backfill_df = fetch_historical_data(
    #     "2023-08-28",
    #     "2026-08-28"
    # )

    # if backfill_df is not None:

    #     save_raw_dataset(
    #         backfill_df
    #     )


    # ========================================================
    # CURRENT DATA UPDATE
    # ========================================================

    print(
        "\nFetching current weather + AQ data..."
    )

    new_data = fetch_current_window()

    if new_data is not None:

        save_raw_dataset(
            new_data
        )

    else:

        print(
            "\nNo data saved during this run."
        )

    print(
        "\nData fetching completed."
    )