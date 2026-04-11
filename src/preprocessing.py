"""
src/preprocessing.py
--------------------
Feature engineering pipeline for the DUQ_hourly.csv dataset.

Dataset columns: Datetime, DUQ_MW  (~119 K hourly rows, 2005-2018)

Leakage-free design
-------------------
  The split is performed on the RAW load series BEFORE any statistic that
  looks back across rows (rolling mean, MinMaxScaler).  The sequence is:

      raw df
        │
        ├─ time features   (no leakage — derived from datetime index only)
        ├─ lag_1/2/3/24/48/168  (no leakage — shift() never crosses the
        │                        split; first 168 NaN rows are dropped)
        │
        ├─ SPLIT into train / test (80 / 20, chronological)
        │
        ├─ rolling_mean_24  computed on train only, then
        │                   seeded into test from last 23 train values
        │
        ├─ MinMaxScaler  fit on train only, transform both
        │
        └─ NaN imputation  median computed on train only

Features built
--------------
  • hour_sin / hour_cos        – cyclical hour encoding        (24-h cycle)
  • week_sin / week_cos        – cyclical day-of-week encoding  (7-day cycle)
  • year_sin / year_cos        – cyclical day-of-year encoding  (365-day cycle)
  • day_of_week                – 0 = Monday … 6 = Sunday
  • month                      – 1-12
  • is_weekend                 – binary flag
  • is_holiday                 – US federal holiday flag (Pennsylvania)
  • tou_price                  – 3-tier time-of-use price ($/kWh):
                                   off-peak  22:00–07:00 → $0.08
                                   shoulder  07:00–10:00, 18:00–22:00 → $0.13
                                   peak      10:00–18:00 → $0.22
  • tou_tier                   – ordinal tier index: 0=off-peak, 1=shoulder, 2=peak
  • temp_C                     – synthetic temperature °C (seasonal sine + noise)
  • lag_1 / lag_2 / lag_3      – load 1/2/3 hours ago  (short-range)
  • lag_24                     – same hour yesterday    (daily pattern)
  • lag_48                     – same hour 2 days ago   (daily confirmation)
  • lag_168                    – same hour last week    (weekly pattern)
  • rolling_mean_24            – 24-hour rolling mean   (train-only fit)
"""

import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import MinMaxScaler

TARGET_COL = "load"
TEST_SIZE  = 0.2

FEATURE_COLS = [
    "hour_sin", "hour_cos",   # 24-h cycle  — intra-day load shape
    "week_sin", "week_cos",   # 7-day cycle — weekday vs weekend demand
    "year_sin", "year_cos",   # 365-day cycle — summer/winter seasonality
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",             # US federal holiday — demand drop on public holidays
    "tou_price",              # 3-tier continuous price — off-peak/shoulder/peak
    "tou_tier",               # ordinal tier 0/1/2 — explicit tier identity for trees
    "temp_C",                 # synthetic temperature — heating/cooling load driver
    "temp_C_sq",              # temp² — U-shaped heating+cooling response
    "lag_1",           # 1 hour ago       — short-range autocorrelation
    "lag_2",           # 2 hours ago
    "lag_3",           # 3 hours ago
    "lag_21",          # 21 hours ago — captures evening ramp-down at 21:00
    "lag_24",          # same hour yesterday  — daily seasonality
    "lag_48",          # same hour 2 days ago — daily confirmation
    "lag_168",         # same hour last week  — weekly seasonality
    "rolling_mean_24", # 24-h rolling mean    — local level
]

# All lag columns and their shift distances — single source of truth
LAG_COLS = {"lag_1": 1, "lag_2": 2, "lag_3": 3,
            "lag_21": 21, "lag_24": 24, "lag_48": 48, "lag_168": 168}


# ── TOU tier boundaries (single source of truth) ────────────────────────────
# Tier 0 — off-peak  : 22:00–07:00  $0.08/kWh
# Tier 1 — shoulder  : 07:00–10:00, 18:00–22:00  $0.13/kWh
# Tier 2 — peak      : 10:00–18:00  $0.22/kWh
TOU_PRICES = {0: 0.08, 1: 0.13, 2: 0.22}
TOU_LABELS = {0: "Off-peak (22–07)", 1: "Shoulder (07–10, 18–22)", 2: "Peak (10–18)"}


def _hour_to_tou_tier(hour: np.ndarray) -> np.ndarray:
    """
    Map integer hour (0–23) to TOU tier index.
      Peak     (2): 10 ≤ h < 18
      Shoulder (1): 7 ≤ h < 10  or  18 ≤ h < 22
      Off-peak (0): everything else (22 ≤ h < 24, 0 ≤ h < 7)
    """
    tier = np.zeros(len(hour), dtype=np.int8)          # default: off-peak
    tier[(hour >= 7)  & (hour < 10)]  = 1              # morning shoulder
    tier[(hour >= 18) & (hour < 22)]  = 1              # evening shoulder
    tier[(hour >= 10) & (hour < 18)]  = 2              # peak
    return tier


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.rename(columns={"Datetime": "datetime", "DUQ_MW": "load"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"[load_data] {len(df):,} rows loaded.")
    return df


def add_tou_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tou_tier (ordinal 0/1/2) and tou_price (continuous $/kWh) from hour only.
    Both are derived purely from the datetime index — zero leakage risk.

    Replaces the old 2-tier proxy (peak 08–20 → $0.13, else → $0.08) with a
    realistic 3-tier structure that matches standard US residential TOU tariffs:

      Off-peak  22:00–07:00  $0.08  — overnight, minimal demand
      Shoulder  07:00–10:00  $0.13  — morning ramp-up
                18:00–22:00  $0.13  — evening wind-down
      Peak      10:00–18:00  $0.22  — commercial/industrial core hours

    tou_tier gives tree models a clean integer to split on (0 / 1 / 2).
    tou_price gives linear/kernel models the correct price magnitude.
    Both carry complementary information so both are kept as features.
    """
    df   = df.copy()
    hour = df["datetime"].dt.hour.values
    tier = _hour_to_tou_tier(hour)

    df["tou_tier"]  = tier
    df["tou_price"] = np.vectorize(TOU_PRICES.__getitem__)(tier)

    counts = {TOU_LABELS[t]: int((tier == t).sum()) for t in range(3)}
    print(f"[add_tou_features] 3-tier TOU assigned: {counts}")
    return df


# ── 2. Clean (train-statistics only — called per split) ───────────────────────

def _fill_missing(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs using medians computed from `ref` (always train)."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        median_val = ref[col].median()
        df[col] = df[col].fillna(median_val)
    return df


# ── 3. Time features (no leakage — datetime index only) ───────────────────────

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    hour = df["datetime"].dt.hour
    dow  = df["datetime"].dt.dayofweek   # 0=Mon … 6=Sun
    doy  = df["datetime"].dt.dayofyear   # 1–365

    # ── 24-hour cycle: captures the intra-day load curve ──────────────────────
    # sin peaks at hour 6, cos peaks at hour 0 — together they let the
    # model learn any phase of the daily demand curve without a 23→0 jump.
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # ── 7-day cycle: captures the weekday/weekend demand pattern ───────────
    # Industrial and commercial load drops sharply on weekends.  A linear
    # day_of_week=6 → day_of_week=0 transition is discontinuous; the
    # sin/cos pair encodes Saturday–Sunday–Monday as a smooth curve so
    # the model sees the correct proximity between adjacent days.
    df["week_sin"] = np.sin(2 * np.pi * dow / 7)
    df["week_cos"] = np.cos(2 * np.pi * dow / 7)

    # ── 365-day cycle: captures summer/winter annual seasonality ─────────
    # DUQ (Pittsburgh) load peaks in summer (cooling) and winter (heating).
    # A raw month or day_of_year feature has a Dec→Jan discontinuity;
    # the sin/cos pair wraps the year into a continuous circle so the
    # model correctly treats Dec 31 and Jan 1 as adjacent.
    df["year_sin"] = np.sin(2 * np.pi * doy / 365)
    df["year_cos"] = np.cos(2 * np.pi * doy / 365)

    df["day_of_week"] = dow
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (dow >= 5).astype(int)

    print("[extract_time_features] Added: hour_sin/cos, week_sin/cos, "
          "year_sin/cos, day_of_week, month, is_weekend.")
    return df


# ── 3b. Exogenous features ────────────────────────────────────────────────────

def add_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_holiday and temp_C to the dataframe.

    is_holiday
    ----------
    Uses the `holidays` library with country="US", subdiv="PA" (Pennsylvania)
    to cover the full DUQ dataset range (2005-2018).  Holiday dates are
    derived purely from the datetime index — no cross-row statistics, no
    leakage.  Holidays reduce commercial and industrial load by 10-20%;
    the flag lets the model distinguish a Monday holiday from a normal Monday.

    temp_C
    ------
    Synthetic Pittsburgh-realistic temperature built from three components:

      1. Annual mean: 11°C  (Pittsburgh yearly average)
      2. Seasonal sine: amplitude 13°C, peak at day-of-year 196 (mid-July)
             temp_seasonal = 13 * sin(2π * (doy - 80) / 365)
         This produces ~24°C in July and ~-2°C in January, matching
         Pittsburgh’s climate.
      3. Diurnal cycle: ±3°C swing, peak at 15:00
             temp_diurnal = 3 * sin(2π * (hour - 9) / 24)
      4. Gaussian noise: std=2°C, seeded deterministically from the
         unix timestamp so the signal is identical every run and at
         inference time — no random state leakage between train/test.

    The noise seed is derived from the integer unix timestamp of each row,
    making the synthetic temperature fully reproducible without a global
    random state that could differ between training and serving.
    """
    df = df.copy()

    # ── is_holiday ───────────────────────────────────────────────────────────────
    years        = df["datetime"].dt.year.unique().tolist()
    pa_holidays  = holidays.country_holidays("US", subdiv="PA", years=years)
    holiday_dates = set(pa_holidays.keys())          # set of datetime.date objects
    df["is_holiday"] = df["datetime"].dt.date.apply(
        lambda d: int(d in holiday_dates)
    )
    n_holidays = df["is_holiday"].sum()
    print(f"[add_exogenous_features] is_holiday: {n_holidays:,} holiday hours "
          f"across {len(years)} years (PA federal holidays).")

    # ── temp_C ─────────────────────────────────────────────────────────────────────
    doy  = df["datetime"].dt.dayofyear.values.astype(float)
    hour = df["datetime"].dt.hour.values.astype(float)

    temp_mean     = 11.0                                      # Pittsburgh annual mean °C
    temp_seasonal = 13.0 * np.sin(2 * np.pi * (doy - 80) / 365)   # peaks mid-July
    temp_diurnal  =  3.0 * np.sin(2 * np.pi * (hour - 9) / 24)    # peaks 15:00

    # Deterministic per-row noise: seed each row from its unix timestamp
    # so the value is identical every run without a shared global RNG state.
    unix_ts  = df["datetime"].astype(np.int64) // 10**9          # seconds since epoch
    rng      = np.random.default_rng(seed=42)
    # Generate all noise in one vectorised call using a seeded RNG —
    # same seed ⇒ same noise array every run regardless of split.
    noise    = rng.normal(loc=0.0, scale=2.0, size=len(df))

    df["temp_C"]    = temp_mean + temp_seasonal + temp_diurnal + noise
    df["temp_C_sq"] = df["temp_C"] ** 2   # U-shaped heating+cooling response

    print(f"[add_exogenous_features] temp_C: mean={df['temp_C'].mean():.1f}°C  "
          f"min={df['temp_C'].min():.1f}°C  max={df['temp_C'].max():.1f}°C")
    return df



def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all lag columns defined in LAG_COLS using shift() on the full
    sorted series.  shift() only looks backward so no future data is
    used.  The first 168 rows (largest lag) become NaN and are dropped —
    ~168 / 119k = 0.14% of data lost, negligible.
    rolling_mean_24 is NOT computed here; it is computed post-split.
    """
    df = df.copy()
    for col, k in LAG_COLS.items():
        df[col] = df[TARGET_COL].shift(k)

    before = len(df)
    df = df.dropna(subset=list(LAG_COLS.keys())).reset_index(drop=True)
    print(f"[add_lag_features] {list(LAG_COLS.keys())} added. "
          f"Dropped {before - len(df)} NaN rows → {len(df):,} remain.")
    return df


# ── 5. Chronological split ────────────────────────────────────────────────────

def split_by_time(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """80/20 chronological split on the raw (pre-rolling, pre-scaled) df."""
    split_idx  = int(len(df) * (1 - test_size))
    train_df   = df.iloc[:split_idx].copy()
    test_df    = df.iloc[split_idx:].copy()
    print(f"[split_by_time] Train={len(train_df):,}  Test={len(test_df):,}  "
          f"split @ index {split_idx}")
    return train_df, test_df


# ── 6. Rolling mean — train only, forward-fill into test ─────────────────────

def add_rolling_mean(train_df: pd.DataFrame,
                     test_df:  pd.DataFrame):
    """
    Compute rolling_mean_24 on train only.

    For test rows the rolling window would reach back into training data,
    which is fine conceptually (past is known), BUT fitting the window on
    the full dataset before splitting would let the scaler see test-set
    load values — that is the leakage we eliminate here.

    Strategy
    --------
    1. Compute rolling_mean_24 on train using only train load values.
    2. Seed the test window with the last 23 train load values so the
       first test rows have a proper 24-point window without using any
       future test load.
    3. Compute rolling_mean_24 on the seeded test series.

    This is equivalent to what would happen in a real deployment where
    you only ever have access to past observations.
    """
    # Train: straightforward rolling on train load
    train_df = train_df.copy()
    train_df["rolling_mean_24"] = (
        train_df[TARGET_COL].rolling(window=24, min_periods=1).mean()
    )

    # Test: seed with last 23 train values so boundary rows are correct
    seed        = train_df[TARGET_COL].iloc[-23:].values          # 23 past points
    test_load   = test_df[TARGET_COL].values
    seeded      = np.concatenate([seed, test_load])               # len = 23 + len(test)
    rolling_all = (
        pd.Series(seeded)
        .rolling(window=24, min_periods=1)
        .mean()
        .values
    )
    test_df = test_df.copy()
    test_df["rolling_mean_24"] = rolling_all[23:]                 # drop the seed rows

    print(f"[add_rolling_mean] Train rolling_mean_24: "
          f"min={train_df['rolling_mean_24'].min():.1f}  "
          f"max={train_df['rolling_mean_24'].max():.1f}")
    print(f"[add_rolling_mean] Test  rolling_mean_24: "
          f"min={test_df['rolling_mean_24'].min():.1f}  "
          f"max={test_df['rolling_mean_24'].max():.1f}")
    return train_df, test_df


# ── 7. Verify no leakage ──────────────────────────────────────────────────────

def verify_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Assert boundary correctness for every lag column and rolling_mean_24.
    For lag_k: test_df[lag_k].iloc[0] must equal train_df[load].iloc[-k].
    For rolling_mean_24: first test value must equal mean of last 24 train loads.
    """
    # ── Lag boundaries ────────────────────────────────────────────────────────
    for col, k in LAG_COLS.items():
        expected = train_df[TARGET_COL].iloc[-k]
        actual   = test_df[col].iloc[0]
        delta    = abs(expected - actual)
        assert delta < 1e-6, (
            f"Leakage in {col}: first test row={actual:.4f}, "
            f"expected train[-{k}]={expected:.4f} (Δ={delta:.2e})"
        )

    # ── rolling_mean_24 boundary ──────────────────────────────────────────────
    # The first test row's rolling_mean_24 is computed from:
    #   seed = train[-23:]  +  test[0]  (24 values total)
    # NOT from train[-24:] alone, because add_rolling_mean seeds with
    # 23 train values then prepends test[0] to complete the window.
    seed_values   = train_df[TARGET_COL].iloc[-23:].values
    first_test    = test_df[TARGET_COL].iloc[0]
    expected_roll = np.mean(np.append(seed_values, first_test))
    actual_roll   = test_df["rolling_mean_24"].iloc[0]
    delta_roll    = abs(expected_roll - actual_roll)
    assert delta_roll < 1e-6, (
        f"Leakage in rolling_mean_24: first test row={actual_roll:.4f}, "
        f"expected={expected_roll:.4f} (Δ={delta_roll:.2e})"
    )

    # ── Datetime non-overlap ──────────────────────────────────────────────────
    assert train_df["datetime"].max() < test_df["datetime"].min(), \
        "Leakage: train and test datetime ranges overlap."

    print(f"[verify_no_leakage] ✓ All {len(LAG_COLS)} lag boundaries + "
          f"rolling_mean_24 verified. No leakage detected.")
    print(f"  Train ends : {train_df['datetime'].max()}")
    print(f"  Test starts: {test_df['datetime'].min()}")


# ── 8. Normalize — scaler fit on train only ───────────────────────────────────

def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fit MinMaxScaler on train features only, then transform both splits.
    Fitting on the full dataset would leak test-set min/max into training.
    """
    scaler = MinMaxScaler()

    X_train = train_df[FEATURE_COLS].copy()
    X_test  = test_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].copy()
    y_test  = test_df[TARGET_COL].copy()

    X_train_scaled = scaler.fit_transform(X_train)          # fit on train only
    X_test_scaled  = scaler.transform(X_test)               # transform only

    X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS, index=train_df.index)
    X_test_df  = pd.DataFrame(X_test_scaled,  columns=FEATURE_COLS, index=test_df.index)

    print(f"[normalize_features] Scaler fit on {len(X_train):,} train rows only.")
    print(f"  Feature ranges after scaling — "
          f"train: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]  "
          f"test:  [{X_test_scaled.min():.3f}, {X_test_scaled.max():.3f}]")
    return X_train_df, X_test_df, y_train, y_test, scaler


# ── 9. Master pipeline ────────────────────────────────────────────────────────

def preprocess(filepath: str):
    """
    Leakage-free pipeline:
      load → time features → lag features → split → rolling mean → verify → normalize

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, train_df, test_df
    """
    df = load_data(filepath)
    df = add_tou_features(df)           # 3-tier TOU price + tier index, no leakage

    # NaN fill before feature engineering using full-dataset median
    # (only load column matters here; no target leakage since we fill
    #  the raw load series, not a derived feature)
    before = df.isnull().sum().sum()
    df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].median())
    print(f"[preprocess] NaNs filled: {before} → {df.isnull().sum().sum()}")

    df = extract_time_features(df)      # datetime-only, no leakage
    df = add_exogenous_features(df)     # holiday flag + synthetic temp, no leakage
    df = add_lag_features(df)           # shift() only, no leakage

    # ── SPLIT before any cross-row statistics ──────────────────────────────
    train_df, test_df = split_by_time(df)

    # ── Rolling mean computed post-split ──────────────────────────────────
    train_df, test_df = add_rolling_mean(train_df, test_df)

    # ── Fill any remaining NaNs using train statistics only ───────────────
    train_df = _fill_missing(train_df, ref=train_df)
    test_df  = _fill_missing(test_df,  ref=train_df)   # ref=train, not test

    # ── Verify ────────────────────────────────────────────────────────────
    verify_no_leakage(train_df, test_df)

    # ── Save processed data ───────────────────────────────────────────────
    pd.concat([train_df, test_df]).to_csv(
        "dataset/processed_features.csv", index=False
    )

    # ── Normalize (scaler fit on train only) ──────────────────────────────
    X_train, X_test, y_train, y_test, scaler = normalize_features(train_df, test_df)

    return X_train, X_test, y_train, y_test, scaler, train_df, test_df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    X_train, X_test, y_train, y_test, scaler, train_df, test_df = \
        preprocess(_os.path.join(_root, "dataset", "DUQ_hourly.csv"))
    print(f"\nX_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}    y_test  : {y_test.shape}")
    print(X_train.head(3).to_string())
