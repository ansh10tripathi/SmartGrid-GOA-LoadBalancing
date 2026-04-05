import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Grid Dashboard",
    layout="wide",
    page_icon="⚡"
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH   = "models/load_forecast_model.pkl"
SCALER_PATH  = "models/minmax_scaler.pkl"

model = None
minmax_scaler = None
if os.path.exists(MODEL_PATH):
    loaded = joblib.load(MODEL_PATH)
    model  = loaded["model"] if isinstance(loaded, dict) else loaded

if os.path.exists(SCALER_PATH):
    minmax_scaler = joblib.load(SCALER_PATH)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚡ Smart Grid Panel")

section = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "📈 Model Analysis",
        "⚙️ Optimization",
        "🔮 Live Prediction",
        "📂 Dataset",
        "📊 All Graphs"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Ansh 🚀")

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("⚡ Smart Grid Load Forecasting & Optimization")

# ─────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────
if section == "📊 Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", "0.9943")
    col2.metric("RMSE", "22.05")
    col3.metric("MAE", "16.21")

    st.success("🏆 Best Model: SVR")

# ─────────────────────────────────────────────
# MODEL ANALYSIS
# ─────────────────────────────────────────────
elif section == "📈 Model Analysis":

    if os.path.exists("results/feature_importance.png"):
        st.image("results/feature_importance.png", width="stretch")

    if os.path.exists("results/actual_vs_predicted.png"):
        st.image("results/actual_vs_predicted.png", width="stretch")

    if os.path.exists("results/model_comparison.png"):
        st.image("results/model_comparison.png", width="stretch")

# ─────────────────────────────────────────────
# OPTIMIZATION
# ─────────────────────────────────────────────
elif section == "⚙️ Optimization":

    if os.path.exists("results/goa_comparison.png"):
        st.image("results/goa_comparison.png", width="stretch")

    if os.path.exists("results/goa_convergence.png"):
        st.image("results/goa_convergence.png", width="stretch")

    if os.path.exists("results/cost_comparison.png"):
        st.image("results/cost_comparison.png", width="stretch")

    if os.path.exists("results/performance_comparison.png"):
        st.image("results/performance_comparison.png", width="stretch")

# ─────────────────────────────────────────────
# 🔮 LIVE PREDICTION (NEW 🔥)
# ─────────────────────────────────────────────
elif section == "🔮 Live Prediction":

    st.subheader("🔮 Real-Time Load Prediction")

    col1, col2 = st.columns(2)

    FEATURE_COLS = [
        "hour_sin", "hour_cos", "day_of_week", "month",
        "is_weekend", "tou_price",
        "lag_1", "lag_2", "lag_3", "rolling_mean_24"
    ]

    col1, col2 = st.columns(2)
    hour        = col1.slider("Hour of Day", 0, 23, 12)
    day_of_week = col1.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    month       = col2.slider("Month", 1, 12, 6)
    lag_1       = col2.number_input("Last Hour Load (MW)", value=1500.0, step=10.0)

    hour_sin        = np.sin(2 * np.pi * hour / 24)
    hour_cos        = np.cos(2 * np.pi * hour / 24)
    is_weekend      = int(day_of_week >= 5)
    tou_price       = 0.13 if 8 <= hour <= 20 else 0.08
    lag_2           = lag_1
    lag_3           = lag_1
    rolling_mean_24 = lag_1

    raw_input = pd.DataFrame([[
        hour_sin, hour_cos, day_of_week, month,
        is_weekend, tou_price,
        lag_1, lag_2, lag_3, rolling_mean_24
    ]], columns=FEATURE_COLS)

    if model is not None and minmax_scaler is not None:
        input_df = pd.DataFrame(
            minmax_scaler.transform(raw_input), columns=FEATURE_COLS
        )
        if isinstance(loaded, dict) and loaded.get("type") == "svr":
            prediction = loaded["model"].predict(loaded["scaler"].transform(input_df))[0]
        else:
            prediction = model.predict(input_df)[0]
        st.success(f"⚡ Predicted Load: {prediction:.2f} MW")
    elif minmax_scaler is None:
        st.warning("MinMax scaler not found. Re-run `python main.py` to generate it.")
    else:
        st.error("Model not loaded!")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
elif section == "📂 Dataset":

    if os.path.exists("dataset/processed_features.csv"):
        df = pd.read_csv("dataset/processed_features.csv")

        st.write(f"Dataset Shape: {df.shape}")
        st.dataframe(df.head(100), width="stretch")

# ─────────────────────────────────────────────
# ALL GRAPHS
# ─────────────────────────────────────────────
elif section == "📊 All Graphs":

    images = sorted([
        f for f in os.listdir("results")
        if f.endswith(".png")
    ])

    cols = st.columns(2)

    for i, img in enumerate(images):
        with cols[i % 2]:
            st.image(f"results/{img}", caption=img, width="stretch")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Smart Grid AI System | Live Prediction + Optimization 🚀")