import pandas as pd
import numpy as np
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH

def load_raw_data():

    print("="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)

    print("\n[1.1] Loading raw data...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, delimiter=';')
        print(f"    ✓ Loaded {len(df):,} records")
        print(f"    ✓ Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"    ✗ Error: File not found at {RAW_DATA_PATH}")
        print("    → Please ensure sensor-fault-detection.csv is in the data/ folder")
        sys.exit(1)

def inspect_data(df):

    print("\n[1.2] Data Quality Inspection...")


    print("\n    Data Types:")
    for col in df.columns:
        print(f"      • {col}: {df[col].dtype}")


    print("\n    Null Values:")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls == 0:
        print("      ✓ No null values found")
    else:
        for col in df.columns:
            if null_counts[col] > 0:
                pct = (null_counts[col] / len(df)) * 100
                print(f"      ⚠️  {col}: {null_counts[col]:,} nulls ({pct:.2f}%)")


    print("\n    SensorId Analysis:")
    unique_sensors = df['SensorId'].nunique()
    print(f"      • Unique SensorId values: {unique_sensors}")
    print(f"      • SensorId value(s): {df['SensorId'].unique()}")

    if unique_sensors == 1:
        print(f"      → SensorId is constant ({df['SensorId'].iloc[0]}), can be dropped")

    return null_counts

def drop_sensor_id(df):

    print("\n[1.3] Dropping SensorId column...")

    original_cols = list(df.columns)
    df_cleaned = df.drop('SensorId', axis=1)

    print(f"    ✓ Dropped SensorId column")
    print(f"    ✓ Remaining columns: {list(df_cleaned.columns)}")

    return df_cleaned

def handle_null_values(df):

    print("\n[1.4] Handling null values...")

    original_count = len(df)
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls == 0:
        print("    ✓ No null values to remove")
        return df


    df_cleaned = df.dropna()
    removed_count = original_count - len(df_cleaned)

    print(f"    ✓ Removed {removed_count:,} rows with null values")
    print(f"    ✓ Remaining records: {len(df_cleaned):,}")

    return df_cleaned

def save_cleaned_data(df):

    print("\n[1.5] Saving cleaned data...")


    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)


    df.to_csv(CLEANED_DATA_PATH, index=False)

    file_size = os.path.getsize(CLEANED_DATA_PATH) / 1024  
    print(f"    ✓ Saved to: {CLEANED_DATA_PATH}")
    print(f"    ✓ File size: {file_size:.2f} KB")
    print(f"    ✓ Records: {len(df):,}")
    print(f"    ✓ Columns: {list(df.columns)}")

def main():


    df = load_raw_data()


    null_counts = inspect_data(df)


    df_cleaned = drop_sensor_id(df)


    df_cleaned = handle_null_values(df_cleaned)


    save_cleaned_data(df_cleaned)

    print("\n" + "="*70)
    print("DATA CLEANING COMPLETE")
    print("="*70)
    print(f"\n✓ Cleaned dataset ready: {len(df_cleaned):,} records")
    print(f"✓ Columns: {list(df_cleaned.columns)}")
    print("\nNext Step:")
    print("  → Run 'python src/data_standardization.py' to standardize the data")
    print("="*70)

if __name__ == "__main__":
    main()
