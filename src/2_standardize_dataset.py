import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLEANED_DATA_PATH, STANDARDIZED_DATA_PATH, SCALER_PATH

def load_cleaned_data():

    print("="*70)
    print("STEP 2: DATA STANDARDIZATION")
    print("="*70)

    print("\n[2.1] Loading cleaned data...")
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        print(f"    ✓ Loaded {len(df):,} records")
        print(f"    ✓ Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"    ✗ Error: Cleaned data not found at {CLEANED_DATA_PATH}")
        print("    → Run 'python src/data_clean.py' first")
        sys.exit(1)

def process_timestamps(df):

    print("\n[2.2] Processing timestamps...")


    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    print(f"    ✓ Converted Timestamp to datetime")
    print(f"    ✓ Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")


    df_sorted = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"    ✓ Data sorted chronologically")

    return df_sorted

def analyze_value_distribution(df):

    print("\n[2.3] Analyzing value distribution...")

    values = df['Value']

    print(f"\n    Original Value Statistics:")
    print(f"      • Count:  {len(values):,}")
    print(f"      • Mean:   {values.mean():.2f}°C")
    print(f"      • Std:    {values.std():.2f}°C")
    print(f"      • Min:    {values.min():.2f}°C")
    print(f"      • 25%:    {values.quantile(0.25):.2f}°C")
    print(f"      • Median: {values.median():.2f}°C")
    print(f"      • 75%:    {values.quantile(0.75):.2f}°C")
    print(f"      • Max:    {values.max():.2f}°C")

def standardize_values(df):

    print("\n[2.4] Standardizing sensor values...")


    values = df[['Value']].values


    scaler = StandardScaler()
    values_standardized = scaler.fit_transform(values)


    df['Value_Standardized'] = values_standardized

    print(f"    ✓ Standardization complete")
    print(f"\n    Standardized Statistics:")
    print(f"      • Mean:   {values_standardized.mean():.6f} (should be ~0)")
    print(f"      • Std:    {values_standardized.std():.6f} (should be ~1)")
    print(f"      • Min:    {values_standardized.min():.2f}")
    print(f"      • Max:    {values_standardized.max():.2f}")


    print(f"\n[2.5] Saving scaler for future use...")
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"    ✓ Scaler saved to: {SCALER_PATH}")

    return df, scaler

def save_standardized_data(df):

    print(f"\n[2.6] Saving standardized data...")


    df.to_csv(STANDARDIZED_DATA_PATH, index=False)

    file_size = os.path.getsize(STANDARDIZED_DATA_PATH) / 1024  
    print(f"    ✓ Saved to: {STANDARDIZED_DATA_PATH}")
    print(f"    ✓ File size: {file_size:.2f} KB")
    print(f"    ✓ Records: {len(df):,}")
    print(f"    ✓ Columns: {list(df.columns)}")

def main():


    df = load_cleaned_data()


    df = process_timestamps(df)


    analyze_value_distribution(df)


    df, scaler = standardize_values(df)


    save_standardized_data(df)

    print("\n" + "="*70)
    print("DATA STANDARDIZATION COMPLETE")
    print("="*70)
    print(f"\n✓ Standardized dataset ready: {len(df):,} records")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Scaler saved for ML model")
    print("\nNext Steps:")
    print("  1. Run 'python src/data_analysis.py' to visualize and explore data")
    print("  2. Run 'python src/train_model.py' to train the ML model")
    print("="*70)

if __name__ == "__main__":
    main()
