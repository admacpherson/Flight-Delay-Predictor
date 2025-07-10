import pandas as pd
import os
import numpy as np
import holidays

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
COLUMNS_TO_KEEP = [
    "FL_DATE",           # Flight date
    "OP_UNIQUE_CARRIER", # Airline code
    "ORIGIN",            # Origin airport
    "DEST",              # Destination airport
    "CRS_DEP_TIME",      # Scheduled departure time
    "DEP_TIME",          # Actual departure time
    "DEP_DELAY",         # Departure delay (mins)
    "ARR_DELAY",         # Arrival delay (mins)
    "CANCELLED"          # Cancellation flag
]

def preprocess_file(filename):
    print(f"Preprocessing {filename}")
    df = pd.read_csv(os.path.join(RAW_DIR, filename))

    # Keep only selected columns
    df = df[COLUMNS_TO_KEEP]

    # Drop canceled flights
    df = df[df["CANCELLED"] == 0]

    # Drop rows with missing delay info
    df = df.dropna(subset=["DEP_DELAY", "ARR_DELAY"])

    # Convert date column to datetime
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="%Y-%m-%d", errors="coerce")

    # Limit delays to a reasonable range
    df = df[(df["DEP_DELAY"] > -60) & (df["DEP_DELAY"] < 360)]

    # Feature engineering: day of week, dep time sin/cos, target label
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek
    df["DEP_HOUR"] = df["CRS_DEP_TIME"] // 100
    df["DEP_MINUTE"] = df["CRS_DEP_TIME"] % 100
    df["DEP_TIME_SIN"] = np.sin(2 * np.pi * (df["DEP_HOUR"] * 60 + df["DEP_MINUTE"]) / (24 * 60))
    df["DEP_TIME_COS"] = np.cos(2 * np.pi * (df["DEP_HOUR"] * 60 + df["DEP_MINUTE"]) / (24 * 60))

    # Binary target: arrival delay > 15 mins
    df["ARR_DEL15"] = (df["ARR_DELAY"] > 15).astype(int)

    # Airport congestion proxy
    df["ORIGIN_FLIGHT_COUNT"] = df.groupby(["FL_DATE", "ORIGIN"])["FL_DATE"].transform("count")

    # Check if holiday
    us_holidays = holidays.US()
    df["IS_HOLIDAY"] = df["FL_DATE"].apply(lambda date: int(date in us_holidays))

    # Save cleaned and engineered dataset
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_file = os.path.join(PROCESSED_DIR, f"cleaned_{filename}")
    df.to_csv(output_file, index=False)
    print(f"Saved cleaned file: {output_file}")

def preprocess_all():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".csv"):
            preprocess_file(filename)

if __name__ == "__main__":
    preprocess_all()