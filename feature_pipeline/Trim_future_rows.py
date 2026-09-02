import pandas as pd
import sys
from pathlib import Path

# Project root = parent of feature_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else PROJECT_ROOT / "data" / "raw_dataset" / "karachi_processed.csv"
)

df = pd.read_csv(FILE)

df["time"] = pd.to_datetime(df["time"])

before = len(df)

now = pd.Timestamp.now().floor("h")

df = df[df["time"] <= now].reset_index(drop=True)

after = len(df)

df.to_csv(FILE, index=False)

print("=" * 60)
print("TRIMMING FUTURE ROWS")
print("=" * 60)
print(f"File: {FILE}")
print(f"Current time: {now}")
print(f"Removed: {before - after} future rows")
print(f"Rows before: {before}")
print(f"Rows after: {after}")
print(f"New date range: {df['time'].min()} -> {df['time'].max()}")
print("=" * 60)