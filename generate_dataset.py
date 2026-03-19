"""
generate_dataset.py
Run this once to create dataset/smartgrid.csv
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 1000

# Hourly timestamps starting Jan 2023
dates = pd.date_range(start="2023-01-01", periods=n, freq="H")
hour  = dates.hour.values
month = dates.month.values
day   = dates.dayofweek.values          # 0=Mon … 6=Sun

# ── Load (kWh) ──────────────────────────────────────────────────────────────
# Two daily peaks: morning (8-10h) and evening (18-20h)
morning_peak = np.exp(-0.5 * ((hour - 9)  / 2) ** 2)
evening_peak = np.exp(-0.5 * ((hour - 19) / 2) ** 2)
seasonal     = 1 + 0.15 * np.sin(2 * np.pi * (month - 1) / 12)   # summer high
weekend_dip  = np.where(day >= 5, 0.85, 1.0)

base_load = 200 + 120 * morning_peak + 100 * evening_peak
load = base_load * seasonal * weekend_dip + np.random.normal(0, 12, n)
load = np.clip(load, 80, 450).round(2)

# ── Price ($/kWh) ────────────────────────────────────────────────────────────
# Time-of-use: peak 08-20h costs more
peak_hours = ((hour >= 8) & (hour <= 20)).astype(float)
price = 0.08 + 0.05 * peak_hours + 0.01 * np.random.rand(n)
price = price.round(4)

# ── Temperature (°C) ─────────────────────────────────────────────────────────
temperature = (
    15
    + 10 * np.sin(2 * np.pi * (month - 3) / 12)   # seasonal
    + 4  * np.sin(2 * np.pi * hour / 24)            # diurnal
    + np.random.normal(0, 1.5, n)
).round(2)

df = pd.DataFrame({
    "datetime":    dates,
    "load":        load,
    "price":       price,
    "temperature": temperature,
})

os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/smartgrid.csv", index=False)
print(f"Dataset saved → dataset/smartgrid.csv  ({len(df)} rows)")
print(df.head(5).to_string())
