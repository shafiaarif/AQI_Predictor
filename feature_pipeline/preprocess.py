import pandas as pd
import os


# PROJECT PATHS

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

# Raw data directory
RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw_dataset"
)

# Input file

RAW_FILE = os.path.join(
    RAW_DIR,
    "karachi_raw.csv"
)

# Output file
PROCESSED_FILE = os.path.join(
    RAW_DIR,
    "karachi_processed.csv"
)

def main():

    print("=" * 60)
    print("KARACHI AQI PREPROCESSING")
    print("=" * 60)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nRaw file:")
    print(RAW_FILE)

    print("\nProcessed file:")
    print(PROCESSED_FILE)

    if not os.path.exists(RAW_FILE):

        raise FileNotFoundError(
            f"\nRaw dataset not found:\n"
            f"{RAW_FILE}\n\n"
            f"Please make sure karachi_raw.csv exists inside:\n"
            f"{RAW_DIR}"
        )

    # LOAD RAW DATA

    df = pd.read_csv(
        RAW_FILE
    )

    print("\n" + "=" * 60)
    print("RAW DATA LOADED")
    print("=" * 60)

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "Column names:"
    )

    print(
        list(df.columns)
    )

    # CHECK TIME COLUMN

    if "time" not in df.columns:

        raise KeyError(
            "The raw dataset does not contain a 'time' column."
        )

    # PARSE TIME

    print("\nParsing timestamps...")

    df["time"] = pd.to_datetime(
        df["time"],
        format="mixed",
        errors="coerce"
    )

    # CHECK INVALID TIMESTAMPS

    invalid_times = (
        df["time"]
        .isna()
        .sum()
    )

    print(
        "Invalid timestamps:",
        invalid_times
    )

    if invalid_times > 0:

        print(
            f"Removing {invalid_times} rows "
            f"with invalid timestamps."
        )

        df = df.dropna(
            subset=["time"]
        )

    # DATA TYPES

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    print(
        df.dtypes
    )

    # MISSING VALUES

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing_values = df.isnull().sum()

    print(
        missing_values
    )

    total_missing = (
        missing_values.sum()
    )

    print(
        "\nTotal missing values:",
        total_missing
    )

    # FULL ROW DUPLICATES

    full_dupes = (
        df.duplicated()
        .sum()
    )

    print(
        "\nFully duplicated rows:",
        full_dupes
    )

    if full_dupes > 0:

        df = df.drop_duplicates()

        print(
            "Full duplicates removed:",
            full_dupes
        )

    # DUPLICATE TIMESTAMPS
    time_dupes = (
        df["time"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate timestamps:",
        time_dupes
    )

    if time_dupes > 0:

        df = df.drop_duplicates(
            subset="time",
            keep="last"
        )

        print(
            "Duplicate timestamps removed:",
            time_dupes
        )

    # SORT CHRONOLOGICALLY

    df = (
        df
        .sort_values(
            by="time"
        )
        .reset_index(
            drop=True
        )
    )

    # CHECK HOURLY CONTINUITY
    print("\n" + "=" * 60)
    print("TIME SERIES CHECK")
    print("=" * 60)

    if len(df) > 1:

        time_diff = (
            df["time"]
            .diff()
            .dropna()
        )

        expected_frequency = pd.Timedelta(
            hours=1
        )

        gaps = (
            time_diff
            != expected_frequency
        ).sum()

        print(
            "Non-hourly gaps:",
            gaps
        )

        if gaps > 0:

            print(
                "WARNING: The dataset contains "
                "non-hourly gaps."
            )

        else:

            print(
                "Hourly continuity: PASS"
            )

    # FINAL DATASET INFORMATION

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "Start:",
        df["time"].min()
    )

    print(
        "End:",
        df["time"].max()
    )

    print(
        "Duplicate timestamps remaining:",
        df["time"].duplicated().sum()
    )

    print(
        "Missing values remaining:",
        df.isnull().sum().sum()
    )

    # CREATE OUTPUT DIRECTORY

    os.makedirs(
        RAW_DIR,
        exist_ok=True
    )

    # SAVE PROCESSED DATASET

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        "Rows after cleaning:",
        len(df)
    )

    print(
        "Date range:",
        df["time"].min(),
        "→",
        df["time"].max()
    )

    print(
        "Saved to:",
        PROCESSED_FILE
    )

    print("=" * 60)

if __name__ == "__main__":
    main()