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
MODEL_PATH         = "models/load_forecast_model.pkl"
SCALER_PATH        = "models/minmax_scaler.pkl"
TARGET_SCALER_PATH = "models/target_scaler.pkl"

# Model is now always a sklearn Pipeline — no dict unwrapping needed
pipeline      = joblib.load(MODEL_PATH)         if os.path.exists(MODEL_PATH)         else None
minmax_scaler = joblib.load(SCALER_PATH)        if os.path.exists(SCALER_PATH)        else None
target_scaler = joblib.load(TARGET_SCALER_PATH) if os.path.exists(TARGET_SCALER_PATH) else None

# ── Single source of truth: load model_results.json once at startup ──────────
import json as _json
from io import BytesIO as _BytesIO
_RESULTS_JSON = "results/model_results.json"
_model_results = None
if os.path.exists(_RESULTS_JSON):
    with open(_RESULTS_JSON, encoding="utf-8") as _f:
        _model_results = _json.load(_f)

# ── paper_table.csv for styled table display ──────────────────────────────────
_PAPER_CSV = "results/paper_table.csv"
if os.path.exists(_PAPER_CSV):
    _perf_df       = pd.read_csv(_PAPER_CSV, index_col=0)
    _best_by_r2    = _perf_df["R²"].idxmax()
    _best_by_rmse  = _perf_df["RMSE (MW)"].idxmin()
    _best_by_mae   = _perf_df["MAE (MW)"].idxmin()
    _best_model    = _best_by_r2
    _best_row      = _perf_df.loc[_best_model]
elif _model_results is not None:
    # Build _perf_df from model_results.json - all 5 models, None-safe
    _rows = {}
    for name, m in _model_results.items():
        if name.startswith("_"):
            continue
        _rows[name] = {
            "RMSE (MW)": m.get("RMSE"),
            "MAE (MW)":  m.get("MAE"),
            "R²":        m.get("R2"),
            "MAPE (%)": m.get("MAPE"),
        }
    _perf_df = pd.DataFrame(_rows).T
    for _c in _perf_df.columns:
        _perf_df[_c] = pd.to_numeric(_perf_df[_c], errors="coerce")
    _perf_df   = _perf_df.dropna(subset=["R²"])
    _best_model   = _perf_df["R²"].idxmax()
    _best_by_r2   = _best_model
    _best_by_rmse = _perf_df["RMSE (MW)"].dropna().idxmin()
    _best_by_mae  = _perf_df["MAE (MW)"].dropna().idxmin()
    _best_row     = _perf_df.loc[_best_model]
else:
    _perf_df = _best_row = None
    _best_model = _best_by_r2 = _best_by_rmse = _best_by_mae = "XGBoost"


def _winner_cards(df: pd.DataFrame) -> None:
    """Render three side-by-side winner cards — one per metric."""
    # dropna per column so None (e.g. QuantileGBR MAPE) doesn't crash idxmax/idxmin
    def _best_model_for(col, higher):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return "N/A", float("nan")
        idx = s.idxmax() if higher else s.idxmin()
        return idx, s[idx]

    winners = [
        ("R²",        True,  "{:.4f}",   "#1a6b3c", "#d4f5e2", "↑ higher is better"),
        ("RMSE (MW)", False, "{:.4f} MW", "#7b3f00", "#fdebd0", "↓ lower is better"),
        ("MAE (MW)",  False, "{:.4f} MW", "#1a3a6b", "#d6eaf8", "↓ lower is better"),
    ]
    cols = st.columns(3)
    for col, (metric, higher, fmt, text_col, bg_col, hint) in zip(cols, winners):
        if metric not in df.columns:
            continue
        model, value = _best_model_for(metric, higher)
        val_str = fmt.format(value) if not (isinstance(value, float) and np.isnan(value)) else "N/A"
        col.markdown(
            f"""
            <div style="
                background:{bg_col}; border-radius:10px; padding:16px 18px;
                border-left:5px solid {text_col}; margin-bottom:4px;
            ">
                <div style="font-size:11px; color:{text_col}; font-weight:600;
                            letter-spacing:0.05em; text-transform:uppercase;
                            margin-bottom:4px;">Best by {metric}</div>
                <div style="font-size:22px; font-weight:700; color:{text_col};
                            margin-bottom:2px;">{model}</div>
                <div style="font-size:15px; color:{text_col}; opacity:0.85;">
                    {metric} = {val_str}</div>
                <div style="font-size:10px; color:{text_col}; opacity:0.6;
                            margin-top:4px;">{hint}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        "🎯 Uncertainty Bounds",
        "🔍 SHAP Explainability",
        "📄 Paper Table",
        "🔬 Statistical Analysis",
        "⚖️ Sensitivity Analysis",
        "🏹 Pareto Front",
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

    if _perf_df is not None:
        # ── Top KPI strip (best model's numbers) ──────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best Model (R²)", _best_model)
        k2.metric("R² Score",        f"{_best_row['R²']:.4f}")
        k3.metric("RMSE (MW)",        f"{_best_row['RMSE (MW)']:.2f}")
        k4.metric("MAE (MW)",         f"{_best_row['MAE (MW)']:.2f}")

        st.markdown("---")

        # ── Per-metric winner cards ────────────────────────────────────────
        st.markdown("##### 🏆 Best model per metric")
        _winner_cards(_perf_df)

        st.markdown("---")

        # ── Full comparison mini-table ──────────────────────────────────────
        st.markdown("##### 📊 All models at a glance")
        st.caption("Full results in → 📄 Paper Table")
        fmt = {c: ("{:.4f}" if c == "R²" else "{:.2f}") for c in _perf_df.columns}

        def _ov_highlight(col):
            higher = col.name == "R²"
            numeric = pd.to_numeric(col, errors="coerce").dropna()
            if numeric.empty:
                return ["" for _ in col]
            best = numeric.max() if higher else numeric.min()
            return [
                "background-color: #b8860b; color: #ffffff; font-weight: bold"
                if (pd.notna(v) and abs(float(v) - best) < 1e-9) else ""
                for v in col
            ]

        st.dataframe(
            _perf_df.style
            .format(fmt, na_rep="—")
            .apply(_ov_highlight, axis=0),
            hide_index=False,
            width="stretch",
        )
    else:
        st.info("Run `python src/paper_comparison.py` to populate metrics.")

# ─────────────────────────────────────────────
# MODEL ANALYSIS
# ─────────────────────────────────────────────
elif section == "📈 Model Analysis":

    st.subheader("📈 Model Analysis")

    # ── KPI table from model_results.json ────────────────────────────────────
    if _model_results is not None:
        _rows = {}
        for name, m in _model_results.items():
            if name.startswith("_"):
                continue
            _rows[name] = {
                "RMSE (MW)": m.get("RMSE"),
                "MAE (MW)":  m.get("MAE"),
                "R²":        m.get("R2"),
                "MAPE (%)": m.get("MAPE"),
            }
        _ma_df = pd.DataFrame(_rows).T

        # Determine best model by R² (highest)
        _ma_best_model = _ma_df["R²"].idxmax()

        def _ma_highlight_row(row):
            """Highlight entire row if it's the best model (by R²)."""
            if row.name == _ma_best_model:
                return ["background-color:#1a6b3c; color:#fff; font-weight:bold"] * len(row)
            return [""] * len(row)

        fmt = {c: ("{:.4f}" if c == "R²" else "{:.4f}") for c in _ma_df.columns
               if _ma_df[c].notna().any()}
        st.dataframe(
            _ma_df.style.format(fmt, na_rep="—").apply(_ma_highlight_row, axis=1),
            width="stretch",
        )
        st.caption(f"✓ Best model: **{_ma_best_model}** (highest R²)")
        st.markdown("---")

    # ── Model selector ────────────────────────────────────────────────────────
    _model_tabs = st.tabs([
        "🌲 Random Forest", "⚡ XGBoost", "📐 SVR", "🧠 LSTM", "📊 All Models"
    ])

    # ── Tab 0: Random Forest ──────────────────────────────────────────────────
    with _model_tabs[0]:
        if _model_results and "RandomForest" in _model_results:
            m = _model_results["RandomForest"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",       f"{m['R2']:.4f}")
            c2.metric("RMSE (MW)", f"{m['RMSE']:.4f}")
            c3.metric("MAE (MW)",  f"{m['MAE']:.4f}")
            c4.metric("MAPE (%)",  f"{m['MAPE']:.4f}" if m.get('MAPE') else "—")
        for img in ["feature_importance.png", "actual_vs_predicted.png",
                    "shap_summary_randomforest.png", "shap_waterfall_randomforest.png"]:
            if os.path.exists(f"results/{img}"):
                st.image(f"results/{img}", caption=img, width="stretch")

    # ── Tab 1: XGBoost ────────────────────────────────────────────────────────
    with _model_tabs[1]:
        if _model_results and "XGBoost" in _model_results:
            m = _model_results["XGBoost"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",       f"{m['R2']:.4f}")
            c2.metric("RMSE (MW)", f"{m['RMSE']:.4f}")
            c3.metric("MAE (MW)",  f"{m['MAE']:.4f}")
            c4.metric("MAPE (%)",  f"{m['MAPE']:.4f}" if m.get('MAPE') else "—")
        for img in ["shap_summary_xgboost.png", "shap_waterfall_xgboost.png"]:
            if os.path.exists(f"results/{img}"):
                st.image(f"results/{img}", caption=img, width="stretch")

    # ── Tab 2: SVR ────────────────────────────────────────────────────────────
    with _model_tabs[2]:
        if _model_results and "SVR" in _model_results:
            m = _model_results["SVR"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",       f"{m['R2']:.4f}")
            c2.metric("RMSE (MW)", f"{m['RMSE']:.4f}")
            c3.metric("MAE (MW)",  f"{m['MAE']:.4f}")
            c4.metric("MAPE (%)",  f"{m['MAPE']:.4f}" if m.get('MAPE') else "—")
        for img in ["shap_summary_svr.png", "shap_waterfall_svr.png"]:
            if os.path.exists(f"results/{img}"):
                st.image(f"results/{img}", caption=img, width="stretch")

    # ── Tab 3: LSTM ───────────────────────────────────────────────────────────
    with _model_tabs[3]:
        if _model_results and "LSTM" in _model_results:
            m = _model_results["LSTM"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",       f"{m['R2']:.4f}")
            c2.metric("RMSE (MW)", f"{m['RMSE']:.4f}")
            c3.metric("MAE (MW)",  f"{m['MAE']:.4f}")
            c4.metric("MAPE (%)",  f"{m['MAPE']:.4f}" if m.get('MAPE') else "—")
        for img in ["lstm_training_curve.png", "lstm_vs_ml_comparison.png"]:
            if os.path.exists(f"results/{img}"):
                st.image(f"results/{img}", caption=img, width="stretch")

    # ── Tab 4: All Models ─────────────────────────────────────────────────────
    with _model_tabs[4]:
        for img in ["model_comparison.png", "paper_comparison.png",
                    "quantile_ribbon.png"]:
            if os.path.exists(f"results/{img}"):
                st.image(f"results/{img}", caption=img, width="stretch")

# ─────────────────────────────────────────────
# OPTIMIZATION
# ─────────────────────────────────────────────
elif section == "⚙️ Optimization":

    # ── GOA KPI metrics from model_results.json ───────────────────────────
    if _model_results is not None:
        goa = _model_results.get("_meta", {}).get("goa", {})
        if goa:
            st.subheader("⚡ GOA Optimization Results")
            c1, c2, c3, c4 = st.columns(4)
            peak_pct = goa.get("peak_pct", 0)
            cost_pct = goa.get("cost_pct", 0)
            par_before = goa.get("par_before", 0)
            par_after  = goa.get("par_after",  0)
            var_before = goa.get("var_before", 0)
            var_after  = goa.get("var_after",  0)
            par_pct  = (par_after  - par_before)  / par_before  * 100 if par_before  else 0
            var_pct  = (var_after  - var_before)  / var_before  * 100 if var_before  else 0
            c1.metric("Peak Reduction",  f"{abs(peak_pct):.1f}%",
                      delta=f"{peak_pct:+.1f}%", delta_color="inverse")
            c2.metric("Cost Savings",    f"{abs(cost_pct):.1f}%",
                      delta=f"{cost_pct:+.1f}%", delta_color="inverse")
            c3.metric("PAR Reduction",   f"{abs(par_pct):.1f}%",
                      delta=f"{par_pct:+.1f}%",  delta_color="inverse")
            c4.metric("Variance Reduction", f"{abs(var_pct):.1f}%",
                      delta=f"{var_pct:+.1f}%",  delta_color="inverse")
            st.caption(
                f"Best model used for GOA: **{goa.get('best_model', 'N/A')}** · "
                f"Best fitness: {goa.get('best_fitness', 0):.4f}"
            )
            st.markdown("---")

    if os.path.exists("results/goa_comparison.png"):
        st.image("results/goa_comparison.png", width="stretch")

    if os.path.exists("results/constraint_comparison.png"):
        st.image("results/constraint_comparison.png",
                 caption="Physical Constraints: Ramp Rate, Ceiling, Floor",
                 width="stretch")

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
        "hour_sin", "hour_cos", "week_sin", "week_cos", "year_sin", "year_cos",
        "day_of_week", "month", "is_weekend", "is_holiday", "tou_price", "tou_tier", "temp_C", "temp_C_sq",
        "lag_1", "lag_2", "lag_3", "lag_21", "lag_24", "lag_48", "lag_168",
        "rolling_mean_24"
    ]

    col1, col2 = st.columns(2)
    hour        = col1.slider("Hour of Day", 0, 23, 12)
    day_of_week = col1.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    month       = col2.slider("Month", 1, 12, 6)
    lag_1       = col2.number_input("Last Hour Load (MW)", value=1500.0, step=10.0)
    is_holiday  = int(col1.checkbox("Public Holiday"))
    temp_C      = col2.slider("Temperature (°C)", -15, 40, 15)

    import datetime as _dt
    doy             = _dt.date(2013, month, min(28, month * 2)).timetuple().tm_yday
    hour_sin        = np.sin(2 * np.pi * hour / 24)
    hour_cos        = np.cos(2 * np.pi * hour / 24)
    week_sin        = np.sin(2 * np.pi * day_of_week / 7)
    week_cos        = np.cos(2 * np.pi * day_of_week / 7)
    year_sin        = np.sin(2 * np.pi * doy / 365)
    year_cos        = np.cos(2 * np.pi * doy / 365)
    is_weekend      = int(day_of_week >= 5)
    # 3-tier TOU — mirrors _hour_to_tou_tier() in preprocessing.py
    if 10 <= hour < 18:
        tou_tier, tou_price = 2, 0.22
    elif (7 <= hour < 10) or (18 <= hour < 22):
        tou_tier, tou_price = 1, 0.13
    else:
        tou_tier, tou_price = 0, 0.08
    lag_2           = lag_1
    lag_3           = lag_1
    lag_21          = lag_1
    lag_24          = lag_1
    lag_48          = lag_1
    lag_168         = lag_1
    rolling_mean_24 = lag_1

    raw_input = pd.DataFrame([[
        hour_sin, hour_cos, week_sin, week_cos, year_sin, year_cos,
        day_of_week, month, is_weekend, is_holiday, tou_price, tou_tier, temp_C, temp_C**2,
        lag_1, lag_2, lag_3, lag_21, lag_24, lag_48, lag_168, rolling_mean_24
    ]], columns=FEATURE_COLS)

    # Debug: show raw input vector
    with st.expander("🔍 Debug: raw input vector"):
        st.dataframe(raw_input)

    if pipeline is None:
        st.error("Model not loaded! Run `python main.py` first.")
    elif minmax_scaler is None:
        st.warning("MinMax scaler not found. Re-run `python main.py` to generate it.")
    elif target_scaler is None:
        st.warning("Target scaler not found. Re-run `python main.py` to regenerate models.")
    else:
        # Step 1: scale features with the same MinMaxScaler used during training
        input_scaled = pd.DataFrame(
            minmax_scaler.transform(raw_input), columns=FEATURE_COLS
        )
        print(f"[DEBUG] Feature vector (MinMax-scaled, pre-pipeline):")
        print(input_scaled.to_string())
        print(f"[DEBUG] Feature scaler fitted: {minmax_scaler is not None}")
        print(f"[DEBUG] Target scaler fitted:  {target_scaler is not None}")
        # Step 2: pipeline.predict() applies its internal StandardScaler then the model
        #         -> output is in normalised [0,1] target space
        pred_normalised = pipeline.predict(input_scaled)[0]
        print(f"[DEBUG] Normalised prediction (scaled target space): {pred_normalised:.6f}")
        # Step 3: inverse-transform back to MW using the saved target scaler
        prediction_mw = target_scaler.inverse_transform(
            np.array([[pred_normalised]])
        )[0, 0]
        print(f"[DEBUG] Inverse-transformed prediction (MW): {prediction_mw:.2f}")
        st.success(f"⚡ Predicted Load: {prediction_mw:.1f} MW")
        with st.expander("🔍 Debug: feature vector, scaling info & prediction"):
            st.markdown("**Raw input (unscaled)**")
            st.dataframe(raw_input)
            st.markdown("**MinMax-scaled input (fed to pipeline)**")
            st.dataframe(input_scaled)
            st.write(f"Normalised prediction (target space [0,1]): `{pred_normalised:.6f}`")
            st.write(f"Inverse-transformed to MW: `{prediction_mw:.2f} MW`")
            st.caption(
                "Pipeline flow: raw → MinMaxScaler (feature) → "
                "Pipeline(StandardScaler → model) → normalised target → "
                "target_scaler.inverse_transform → MW"
            )

# ─────────────────────────────────────────────
# UNCERTAINTY BOUNDS
# ─────────────────────────────────────────────
elif section == "🎯 Uncertainty Bounds":

    st.subheader("🎯 Quantile Regression — Prediction Intervals")
    st.caption(
        "Three GradientBoostingRegressor models (q = 0.10 / 0.50 / 0.90) "
        "produce an 80% prediction interval around every forecast. "
        "Run `python src/quantile_model.py` once to generate the data."
    )

    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from src.quantile_model import load_predictions

    data = load_predictions()

    if data is None:
        st.warning(
            "No quantile predictions found. "
            "Run `python src/quantile_model.py` first."
        )
    else:
        # ── KPI row ──────────────────────────────────────────────────────────
        cov   = float(data["coverage"][0])
        width = float(data["width"][0])
        r2    = float(data["r2"][0])
        rmse  = float(data["rmse"][0])
        mae   = float(data["mae"][0])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("80% Coverage",    f"{cov*100:.1f}%",
                  delta="✓ meets target" if cov >= 0.80 else "✗ below 80%",
                  delta_color="normal" if cov >= 0.80 else "inverse")
        c2.metric("Mean Width (MW)", f"{width:.1f}")
        c3.metric("Median R²",       f"{r2:.4f}")
        c4.metric("Median RMSE",     f"{rmse:.2f} MW")
        c5.metric("Median MAE",      f"{mae:.2f} MW")

        st.markdown("---")

        # ── Interactive ribbon plot ───────────────────────────────────────────
        n_total = len(data["y_true"])
        n_hours = st.slider(
            "Hours to display (test set)",
            min_value=48, max_value=min(n_total, 2160),
            value=336, step=24,
            help="336 h = 2 weeks  |  720 h = 1 month"
        )

        idx    = np.arange(n_hours)
        y_true = data["y_true"][:n_hours]
        lower  = data["lower"][:n_hours]
        median = data["median"][:n_hours]
        upper  = data["upper"][:n_hours]

        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(13, 4))
        ax.fill_between(idx, lower, upper,
                        alpha=0.25, color="steelblue",
                        label="80% interval (10–90%)")
        ax.plot(idx, y_true,  color="black",     lw=1.3, label="Actual load")
        ax.plot(idx, median,  color="steelblue", lw=1.0,
                linestyle="--", label="Median forecast")
        ax.plot(idx, lower,   color="steelblue", lw=0.6, linestyle=":")
        ax.plot(idx, upper,   color="steelblue", lw=0.6, linestyle=":")
        ax.set_xlabel("Hour (test set)")
        ax.set_ylabel("Load (MW)")
        ax.set_title(
            f"Quantile Regression Ribbon  "
            f"(coverage={cov*100:.1f}%,  mean width={width:.1f} MW)"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        _plt.tight_layout()
        _buf = _BytesIO()
        fig.savefig(_buf, format='png', dpi=120, bbox_inches='tight')
        _buf.seek(0)
        st.image(_buf)
        _plt.close(fig)

        # ── Coverage breakdown table ──────────────────────────────────────────
        st.markdown("**Per-quantile coverage breakdown**")
        inside   = (y_true >= lower) & (y_true <= upper)
        below_lo = y_true < lower
        above_hi = y_true > upper
        st.dataframe(
            {
                "Band":          ["Below 10th pct", "Inside 80% interval", "Above 90th pct"],
                "Count":         [int(below_lo.sum()), int(inside.sum()), int(above_hi.sum())],
                "Fraction (%)":  [
                    f"{below_lo.mean()*100:.1f}",
                    f"{inside.mean()*100:.1f}",
                    f"{above_hi.mean()*100:.1f}",
                ],
            },
            hide_index=True,
        )

# ─────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
elif section == "🔍 SHAP Explainability":

    st.subheader("🔍 SHAP Model Explainability")
    st.caption(
        "TreeExplainer for RF & XGBoost · KernelExplainer for SVR. "
        "Run `python src/explainability.py` once to generate the data."
    )

    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from src.explainability import load_shap

    data = load_shap()

    if data is None:
        st.warning(
            "No SHAP data found. "
            "Run `python src/explainability.py` first."
        )
    else:
        feature_names = list(data["feature_names"])
        peak_idx      = int(data["peak_idx"][0])

        # Only show models whose SHAP arrays are actually present in the .npz
        _sv_map = {"RandomForest": "rf_sv", "XGBoost": "xgb_sv", "SVR": "svr_sv"}
        _ev_map = {"RandomForest": "rf_ev", "XGBoost": "xgb_ev", "SVR": "svr_ev"}
        _available = [m for m, k in _sv_map.items() if k in data]
        if not _available:
            st.warning("No SHAP arrays found in shap_values.npz. Re-run `python main.py`.")
            st.stop()

        model_choice = st.selectbox("Select model", _available)

        sv_key = _sv_map[model_choice]
        ev_key = _ev_map[model_choice]
        shap_vals      = data[sv_key]          # (n_samples, n_features)
        expected_value = float(data[ev_key][0])
        n_samples      = len(shap_vals)

        st.markdown("---")

        # ── Summary bar chart ────────────────────────────────────────────────
        st.markdown(f"**Mean |SHAP| feature importance — {model_choice}**")
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order    = np.argsort(mean_abs)
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(8, 6))
        bars = ax.barh(
            [feature_names[i] for i in order],
            mean_abs[order],
            color="steelblue", alpha=0.85,
        )
        for bar, val in zip(bars[-5:], mean_abs[order][-5:]):
            ax.text(val + mean_abs.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Mean |SHAP value| (MW)")
        ax.set_title(f"SHAP Feature Importance — {model_choice}  ({n_samples:,} samples)")
        ax.grid(axis="x", alpha=0.3)
        _plt.tight_layout()
        _buf = _BytesIO()
        fig.savefig(_buf, format='png', dpi=120, bbox_inches='tight')
        _buf.seek(0)
        st.image(_buf)
        _plt.close(fig)

        st.markdown("---")

        # ── Waterfall for a single sample ────────────────────────────────────
        max_idx = n_samples - 1
        default = min(peak_idx, max_idx)
        sample_idx = st.slider(
            "Test-set sample index for waterfall",
            min_value=0, max_value=max_idx, value=default,
            help=f"Default = peak load hour (index {default})",
        )

        sv_row = shap_vals[sample_idx]
        pred   = expected_value + sv_row.sum()
        st.markdown(
            f"**Waterfall — {model_choice}** &nbsp;|&nbsp; "
            f"baseline = {expected_value:.1f} MW &nbsp;·&nbsp; "
            f"prediction = {pred:.1f} MW"
        )

        # Top-15 features by |SHAP|
        top_n   = 15
        top_ord = np.argsort(np.abs(sv_row))[-top_n:]  # ascending
        names_w  = [feature_names[i] for i in top_ord]
        vals_w   = sv_row[top_ord]

        running = expected_value
        lefts, heights, bar_colors = [], [], []
        for v in vals_w:
            lefts.append(running if v >= 0 else running + v)
            heights.append(abs(v))
            bar_colors.append("tomato" if v >= 0 else "steelblue")
            running += v

        fig2, ax2 = _plt.subplots(figsize=(9, 5))
        ax2.barh(range(top_n), heights, left=lefts,
                 color=bar_colors, alpha=0.85)
        ax2.axvline(expected_value, color="black",  lw=1.0, ls="--",
                    label=f"E[f(x)] = {expected_value:.1f}")
        ax2.axvline(pred,           color="purple", lw=1.2, ls="-",
                    label=f"f(x) = {pred:.1f}")
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(names_w, fontsize=9)
        ax2.set_xlabel("SHAP value (MW)")
        ax2.set_title(f"SHAP Waterfall — {model_choice}  (sample {sample_idx})")
        ax2.legend(fontsize=8)
        ax2.grid(axis="x", alpha=0.3)
        _plt.tight_layout()
        _buf = _BytesIO()
        fig2.savefig(_buf, format='png', dpi=120, bbox_inches='tight')
        _buf.seek(0)
        st.image(_buf)
        _plt.close(fig2)

        # ── Top-feature table ────────────────────────────────────────────────
        st.markdown("**Top 10 features by mean |SHAP| (test set)**")
        top10_ord = np.argsort(mean_abs)[::-1][:10]
        st.dataframe(
            {
                "Feature":        [feature_names[i] for i in top10_ord],
                "Mean |SHAP| (MW)": [f"{mean_abs[i]:.4f}" for i in top10_ord],
            },
            hide_index=True,
        )

# ─────────────────────────────────────────────
# PAPER TABLE
# ─────────────────────────────────────────────
elif section == "📄 Paper Table":

    st.subheader("📄 Publication-Ready Model Comparison")
    st.caption(
        "RMSE / MAE / R² / MAPE on the DUQ test set. "
        "Run `python src/paper_comparison.py` once to generate the files."
    )

    import os as _os
    _csv  = _os.path.join("results", "paper_table.csv")
    _tex  = _os.path.join("results", "paper_table.tex")
    _png  = _os.path.join("results", "paper_comparison.png")

    # ── Build full 5-model table from model_results.json (primary source) ──
    # paper_table.csv only has RF/XGB/SVR; model_results.json has all 5.
    import pandas as _pd
    if _model_results is not None:
        _pt_rows = {}
        _display_names = {
            "RandomForest": "Random Forest",
            "SVR": "SVR",
            "XGBoost": "XGBoost",
            "LSTM": "LSTM",
            "QuantileGBR": "Quantile GBR (q=0.50)",
        }
        for key, label in _display_names.items():
            if key not in _model_results:
                continue
            m = _model_results[key]
            _pt_rows[label] = {
                "RMSE (MW)": m.get("RMSE"),
                "MAE (MW)":  m.get("MAE"),
                "R²":        m.get("R2"),
                "MAPE (%)": m.get("MAPE"),
            }
        df = _pd.DataFrame(_pt_rows).T
        # best model name in display-name space
        _pt_best = _display_names.get(_best_model, _best_model)
    elif _os.path.exists(_csv):
        df = _pd.read_csv(_csv, index_col=0)
        _pt_best = _best_model
    else:
        st.warning("No comparison data found. Run `python main.py` first.")
        st.stop()

    higher_better = {"R²": True}

    def _highlight_best_row(row):
        if row.name == _pt_best:
            return ["background-color: #b8860b; color: #ffffff; font-weight: bold"] * len(row)
        return [""] * len(row)

    def _highlight_best_cell(col):
        hb      = higher_better.get(col.name, False)
        numeric = _pd.to_numeric(col, errors="coerce").dropna()
        if numeric.empty:
            return ["" for _ in col]
        best = numeric.max() if hb else numeric.min()
        return [
            "border-bottom: 2px solid #ffd700"
            if (_pd.notna(v) and abs(float(v) - best) < 1e-9) else ""
            for v in col
        ]

    fmt = {c: ("{:.4f}" if c == "R²" else "{:.4f}") for c in df.columns}
    styled = (
        df.style
        .apply(_highlight_best_row, axis=1)
        .apply(_highlight_best_cell, axis=0)
        .format(fmt, na_rep="—")
    )
    st.dataframe(styled, width="stretch")

    # ── Per-metric winner cards ──────────────────────────────────────────
    st.markdown("##### 🏆 Best model per metric")
    _winner_cards(df)

    st.markdown("---")

    # ── Grouped bar chart ────────────────────────────────────────────────
    import matplotlib.pyplot as _plt
    _metrics_to_plot = [c for c in ["RMSE (MW)", "MAE (MW)", "R²", "MAPE (%)"] if c in df.columns]
    _colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2"]
    _models = list(df.index)
    _bar_colors = _colors[:len(_models)]
    _fig, _axes = _plt.subplots(1, len(_metrics_to_plot),
                                figsize=(4.5 * len(_metrics_to_plot), 5))
    if len(_metrics_to_plot) == 1:
        _axes = [_axes]
    for _ax, _met in zip(_axes, _metrics_to_plot):
        _vals = _pd.to_numeric(df[_met], errors="coerce").fillna(0).values
        _hb   = higher_better.get(_met, False)
        _best_i = int(np.argmax(_vals) if _hb else np.argmin(_vals))
        _bars = _ax.bar(_models, _vals, color=_bar_colors, alpha=0.85, width=0.5)
        for _i, (_b, _v) in enumerate(zip(_bars, _vals)):
            _ax.text(_b.get_x() + _b.get_width()/2, _b.get_height()*1.005,
                     f"{_v:.4f}", ha="center", va="bottom", fontsize=7,
                     fontweight="bold" if _i == _best_i else "normal")
        _bars[_best_i].set_edgecolor("goldenrod")
        _bars[_best_i].set_linewidth(2)
        _ax.set_title(_met, fontsize=9)
        _ax.set_xticks(range(len(_models)))
        _ax.set_xticklabels(_models, rotation=20, ha="right", fontsize=8)
        _ax.grid(axis="y", alpha=0.3)
    _fig.suptitle("All-Model Performance Comparison (★ = best per metric)",
                  fontsize=11)
    _plt.tight_layout()
    _buf = _BytesIO()
    _fig.savefig(_buf, format='png', dpi=120, bbox_inches='tight')
    _buf.seek(0)
    st.image(_buf)
    _plt.close(_fig)

    st.markdown("---")

    # ── LaTeX source (from file if available) ────────────────────────────
    if _os.path.exists(_tex):
        with open(_tex, encoding="utf-8") as _f:
            latex_src = _f.read()
        with st.expander("📋 Copy LaTeX source"):
            st.code(latex_src, language="latex")
            st.download_button(
                label="⬇️ Download .tex",
                data=latex_src,
                file_name="paper_table.tex",
                mime="text/plain",
            )

# ─────────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────────
elif section == "🔬 Statistical Analysis":

    st.subheader("🔬 Statistical Significance Analysis")
    st.caption(
        "GOA vs PSO, GA, DE over 30 independent runs (seeds 1-30). "
        "Wilcoxon signed-rank test. Run `python statistical_analysis.py` to generate."
    )

    _stat_csv = "results/statistical_comparison.csv"
    _stat_tex = "results/statistical_comparison.tex"

    if not os.path.exists(_stat_csv):
        st.warning("No results found. Run `python statistical_analysis.py` first.")
    else:
        _stat_df = pd.read_csv(_stat_csv)

        # KPI cards — GOA row
        goa_row = _stat_df[_stat_df["Algorithm"].str.contains("GOA")].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GOA Mean Fitness",  f"{goa_row['Mean']:.6f}")
        c2.metric("GOA Std",           f"{goa_row['Std']:.6f}")
        c3.metric("GOA Best",          f"{goa_row['Best']:.6f}")
        c4.metric("GOA Worst",         f"{goa_row['Worst']:.6f}")

        st.markdown("---")

        # Colour Sig column
        def _sig_color(val):
            if val == "**":  return "background-color:#1a6b3c; color:white; font-weight:bold"
            if val == "*":   return "background-color:#7b3f00; color:white"
            return ""

        st.dataframe(
            _stat_df.style.map(_sig_color, subset=["Sig"]),
            hide_index=True, width="stretch",
        )
        st.caption("** p<0.01   * p<0.05   ns = not significant")

        st.markdown("---")

        # Bar chart: mean fitness per algorithm
        import matplotlib.pyplot as _plt
        numeric_df = _stat_df[pd.to_numeric(_stat_df["Mean"], errors="coerce").notna()].copy()
        numeric_df["Mean"] = numeric_df["Mean"].astype(float)
        fig, ax = _plt.subplots(figsize=(8, 4))
        colors = ["gold" if "GOA" in a else "steelblue" for a in numeric_df["Algorithm"]]
        bars = ax.bar(numeric_df["Algorithm"], numeric_df["Mean"],
                      color=colors, alpha=0.85, width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.002,
                    f"{bar.get_height():.5f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Mean Best Fitness (lower = better)")
        ax.set_title("Algorithm Comparison: Mean Fitness over 30 Runs")
        ax.grid(axis="y", alpha=0.3)
        _plt.tight_layout()
        _buf = _BytesIO()
        fig.savefig(_buf, format='png', dpi=120, bbox_inches='tight')
        _buf.seek(0)
        st.image(_buf)
        _plt.close(fig)

        if os.path.exists(_stat_tex):
            with open(_stat_tex, encoding="utf-8") as _f:
                tex_src = _f.read()
            with st.expander("📋 LaTeX Table"):
                st.code(tex_src, language="latex")
                st.download_button("⬇️ Download .tex", tex_src,
                                   "statistical_comparison.tex", "text/plain")

# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────
elif section == "⚖️ Sensitivity Analysis":

    st.subheader("⚖️ Weight Sensitivity Analysis")
    st.caption(
        "Grid search over w_peak, w_cost, w_var in {0.1, 0.3, 0.5, 0.7}. "
        "Run `python sensitivity_analysis.py` to generate."
    )

    _sens_csv = "results/sensitivity_results.csv"
    _sens_tex = "results/sensitivity_analysis.tex"

    if not os.path.exists(_sens_csv):
        st.warning("No results found. Run `python sensitivity_analysis.py` first.")
    else:
        _sens_df = pd.read_csv(_sens_csv)
        pareto_df = _sens_df[_sens_df["pareto"] == True]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Combinations",    len(_sens_df))
        c2.metric("Pareto-Optimal",         len(pareto_df))
        c3.metric("Best Peak Reduction",
                  f"{_sens_df['peak_red_%'].max():.2f}%")

        st.markdown("---")
        st.markdown("**Pareto-optimal weight combinations**")
        st.dataframe(pareto_df[["w_peak","w_cost","w_var","w_par",
                                 "peak_red_%","cost_red_%","var_red_%"]]
                     .reset_index(drop=True),
                     hide_index=True, width="stretch")

        st.markdown("---")
        for fname, caption in [
            ("sensitivity_heatmap_peak_cost.png", "Peak Reduction Heatmap (w_peak vs w_cost)"),
            ("sensitivity_heatmap_cost_var.png",  "Cost Reduction Heatmap"),
            ("sensitivity_heatmap_peak_var.png",  "Variance Reduction Heatmap"),
            ("sensitivity_pareto.png",             "3-D Pareto Scatter"),
        ]:
            path = f"results/{fname}"
            if os.path.exists(path):
                st.image(path, caption=caption, width="stretch")

        if os.path.exists(_sens_tex):
            with open(_sens_tex, encoding="utf-8") as _f:
                tex_src = _f.read()
            with st.expander("📋 LaTeX Table"):
                st.code(tex_src, language="latex")
                st.download_button("⬇️ Download .tex", tex_src,
                                   "sensitivity_analysis.tex", "text/plain")

# ─────────────────────────────────────────────
# PARETO FRONT
# ─────────────────────────────────────────────
elif section == "🏹 Pareto Front":

    st.subheader("🏹 Multi-Objective Pareto Front")
    st.caption(
        "200 GOA runs with Dirichlet-sampled weights. "
        "Objectives: Peak Reduction, Cost Reduction, Variance Reduction. "
        "Run `python pareto_analysis.py` to generate."
    )

    _par_csv = "results/pareto_front.csv"
    _par_tex = "results/pareto_front.tex"
    _par_png = "results/pareto_front.png"

    if not os.path.exists(_par_csv):
        st.warning("No results found. Run `python pareto_analysis.py` first.")
    else:
        _par_df   = pd.read_csv(_par_csv)
        _par_opt  = _par_df[_par_df["pareto"] == True]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs",          len(_par_df))
        c2.metric("Pareto-Optimal",       len(_par_opt))
        c3.metric("Best Peak Red.",       f"{_par_df['peak_red_%'].max():.2f}%")
        c4.metric("Best Cost Red.",       f"{_par_df['cost_red_%'].max():.2f}%")

        st.markdown("---")

        if os.path.exists(_par_png):
            st.image(_par_png, caption="Pareto Front: Peak vs Cost vs Variance",
                     width="stretch")

        st.markdown("**Pareto-optimal solutions**")
        st.dataframe(
            _par_opt[["w_peak","w_cost","w_var",
                       "peak_red_%","cost_red_%","var_red_%"]]
            .sort_values("peak_red_%", ascending=False)
            .reset_index(drop=True),
            hide_index=True, width="stretch",
        )

        if os.path.exists(_par_tex):
            with open(_par_tex, encoding="utf-8") as _f:
                tex_src = _f.read()
            with st.expander("📋 LaTeX Table"):
                st.code(tex_src, language="latex")
                st.download_button("⬇️ Download .tex", tex_src,
                                   "pareto_front.tex", "text/plain")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
elif section == "📂 Dataset":

    st.subheader("📂 Processed Dataset Explorer")

    _csv_path = "dataset/processed_features.csv"

    if not os.path.exists(_csv_path):
        st.warning("Processed dataset not found. Run `python main.py` first.")
    else:
        df = pd.read_csv(_csv_path)
        total_rows = len(df)
        total_cols = len(df.columns)

        # ── Info row ───────────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Total Rows", f"{total_rows:,}")
        c2.metric("📋 Total Columns", total_cols)
        c3.metric("📅 Date Range",
                  f"{pd.to_datetime(df['datetime']).dt.year.min()} – "
                  f"{pd.to_datetime(df['datetime']).dt.year.max()}"
                  if "datetime" in df.columns else "N/A")

        st.markdown("---")

        # ── Controls row ──────────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])

        view_mode = ctrl1.radio(
            "View mode",
            ["First N rows", "Last N rows", "Random sample"],
            horizontal=True,
        )

        n_rows = ctrl2.slider(
            "Number of rows to display",
            min_value=10,
            max_value=min(total_rows, 10_000),
            value=100,
            step=10,
            help=f"Dataset has {total_rows:,} rows total",
        )

        col_filter = ctrl3.multiselect(
            "Filter columns (leave empty = all)",
            options=list(df.columns),
            default=[],
        )

        # ── Apply view mode ─────────────────────────────────────────────────────────
        if view_mode == "First N rows":
            view_df = df.head(n_rows)
        elif view_mode == "Last N rows":
            view_df = df.tail(n_rows)
        else:
            view_df = df.sample(n=min(n_rows, total_rows), random_state=42)

        if col_filter:
            view_df = view_df[col_filter]

        st.caption(
            f"Showing {len(view_df):,} rows × {len(view_df.columns)} columns "
            f"({view_mode.lower()})"
        )
        st.dataframe(view_df, width="stretch", hide_index=True)

        st.markdown("---")

        # ── Download section ─────────────────────────────────────────────────────────
        st.markdown("**⬇️ Download**")
        dl1, dl2 = st.columns(2)

        # Download current view
        dl1.download_button(
            label=f"⬇️ Download current view ({len(view_df):,} rows) as CSV",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name=f"smartgrid_{view_mode.replace(' ', '_').lower()}_{n_rows}.csv",
            mime="text/csv",
        )

        # Download full dataset
        dl2.download_button(
            label=f"⬇️ Download full dataset ({total_rows:,} rows) as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="smartgrid_processed_features_full.csv",
            mime="text/csv",
        )

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