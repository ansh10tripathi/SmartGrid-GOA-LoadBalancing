"""
src/preprocessing.py
--------------------
Feature engineering pipeline for the DUQ_hourly.csv dataset.

Dataset columns: Datetime, DUQ_MW  (~119 K hourly rows, 2005-2018)

Features built
--------------
  • hour_sin / hour_cos   – cyclical hour encoding (avoids 23→0 discontinuity)
  • day_of_week           – 0 = Monday … 6 = Sunday
  • month                 – 1-12
  • is_weekend            – binary flag
  • tou_price             – time-of-use price proxy derived from hour
                            (peak 08-20 h → 0.13 $/kWh, off-peak → 0.08 $/kWh)
  • lag_1 / lag_2 / lag_3 – load 1/2/3 hours ago  (biggest R² driver)
  • rolling_mean_24       – 24-hour rolling mean of load
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

TARGET_COL = "load"

FEATURE_COLS = [
    "hour_sin", "hour_cos",   # cyclical hour encoding
    "day_of_week",            # 0 = Monday … 6 = Sunday
    "month",
    "is_weekend",
    "tou_price",              # time-of-use price proxy ($/kWh)
    "lag_1",                  # load 1 hour ago
    "lag_2",                  # load 2 hours ago
    "lag_3",                  # load 3 hours ago
    "rolling_mean_24",        # 24-hour rolling mean of load
]


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df.rename(columns={
        "Datetime": "datetime",
        "DUQ_MW":   "load"
    }, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Time-of-use price proxy: peak hours (08-20) cost more
    hour = df["datetime"].dt.hour
    df["tou_price"] = np.where((hour >= 8) & (hour <= 20), 0.13, 0.08)

    print(f"[load_data] {len(df):,} rows, {df.shape[1]} columns.")
    return df


# ── 2. Clean ──────────────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = df.isnull().sum().sum()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=[TARGET_COL])
    print(f"[handle_missing] NaNs before={before}, after={df.isnull().sum().sum()}")
    return df.reset_index(drop=True)


# ── 3. Feature Engineering ────────────────────────────────────────────────────

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["datetime"].dt.hour

    df["hour_sin"]   = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    print("[extract_time_features] Added: hour_sin, hour_cos, day_of_week, month, is_weekend")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_2"] = df[TARGET_COL].shift(2)
    df["lag_3"] = df[TARGET_COL].shift(3)

    df["rolling_mean_24"] = (
        df[TARGET_COL].rolling(window=24, min_periods=1).mean()
    )

    before = len(df)
    df = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)
    print(f"[add_lag_features] Added lag_1/2/3 + rolling_mean_24. "
          f"Dropped {before - len(df)} NaN rows -> {len(df):,} remain.")
    return df


# ── 4. Normalize ──────────────────────────────────────────────────────────────

def normalize_features(df: pd.DataFrame):
    scaler = MinMaxScaler()
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=df.index)

    print(f"[normalize_features] Scaled {len(FEATURE_COLS)} features: {FEATURE_COLS}")
    return X_df, y, scaler


# ── 5. Master pipeline ────────────────────────────────────────────────────────

def preprocess(filepath: str):
    """
    Full pipeline: load → clean → time features → lag features → normalize.
    Returns (X, y, scaler, processed_df).
    """
    df = load_data(filepath)
    df = handle_missing(df)
    df = extract_time_features(df)
    df = add_lag_features(df)

    df.to_csv("dataset/processed_features.csv", index=False)
    
    X, y, scaler = normalize_features(df)
    return X, y, scaler, df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    X, y, scaler, df = preprocess(_os.path.join(_root, "dataset", "DUQ_hourly.csv"))
    print(f"\nFeature matrix : {X.shape}")
    print(f"Target vector  : {y.shape}")
    print(X.head(3).to_string())
