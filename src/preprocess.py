import pandas as pd
import os

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

    # Drop rows where flight was canceled
    df = df[df["CANCELLED"] == 0]

    # Drop rows with missing delay info
    df = df.dropna(subset=["DEP_DELAY", "ARR_DELAY"])

    # Convert date column to datetime
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    # Keep only positive and small negative delays (optional)
    df = df[(df["DEP_DELAY"] > -60) & (df["DEP_DELAY"] < 360)]

    # Save cleaned version
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