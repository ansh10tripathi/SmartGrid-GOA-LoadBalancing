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
MODEL_PATH  = "models/load_forecast_model.pkl"
SCALER_PATH = "models/minmax_scaler.pkl"

# Model is now always a sklearn Pipeline — no dict unwrapping needed
pipeline      = joblib.load(MODEL_PATH)  if os.path.exists(MODEL_PATH)  else None
minmax_scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# ── Single source of truth for best model ────────────────────────────────────
# Loaded once at startup; every section reads from this.
_PAPER_CSV = "results/paper_table.csv"
if os.path.exists(_PAPER_CSV):
    _perf_df       = pd.read_csv(_PAPER_CSV, index_col=0)
    _best_by_r2    = _perf_df["R²"].idxmax()
    _best_by_rmse  = _perf_df["RMSE (MW)"].idxmin()
    _best_by_mae   = _perf_df["MAE (MW)"].idxmin()
    _best_model    = _best_by_r2          # primary criterion used for GOA / saved model
    _best_row      = _perf_df.loc[_best_model]
else:
    _perf_df = _best_row = None
    _best_model = _best_by_r2 = _best_by_rmse = _best_by_mae = "XGBoost"


def _winner_cards(df: pd.DataFrame) -> None:
    """Render three side-by-side winner cards — one per metric."""
    winners = [
        ("R²",       df["R²"].idxmax(),       df["R²"].max(),       "{:.4f}",  "#1a6b3c", "#d4f5e2", "↑ higher is better"),
        ("RMSE (MW)", df["RMSE (MW)"].idxmin(), df["RMSE (MW)"].min(), "{:.2f} MW", "#7b3f00", "#fdebd0", "↓ lower is better"),
        ("MAE (MW)",  df["MAE (MW)"].idxmin(),  df["MAE (MW)"].min(),  "{:.2f} MW", "#1a3a6b", "#d6eaf8", "↓ lower is better"),
    ]
    cols = st.columns(3)
    for col, (metric, model, value, fmt, text_col, bg_col, hint) in zip(cols, winners):
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
                    {metric} = {fmt.format(value)}</div>
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
            best   = col.max() if higher else col.min()
            return [
                "background-color: #b8860b; color: #ffffff; font-weight: bold"
                if abs(v - best) < 1e-9 else ""
                for v in col
            ]

        st.dataframe(
            _perf_df.style
            .format(fmt)
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

    if pipeline is not None and minmax_scaler is not None:
        input_df = pd.DataFrame(
            minmax_scaler.transform(raw_input), columns=FEATURE_COLS
        )
        # Pipeline.predict() runs StandardScaler → model internally
        prediction = pipeline.predict(input_df)[0]
        st.success(f"⚡ Predicted Load: {prediction:.2f} MW")
    elif minmax_scaler is None:
        st.warning("MinMax scaler not found. Re-run `python main.py` to generate it.")
    else:
        st.error("Model not loaded!")

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
        st.pyplot(fig)
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

        model_choice = st.selectbox(
            "Select model",
            ["RandomForest", "XGBoost", "SVR"],
        )

        sv_key = {"RandomForest": "rf_sv", "XGBoost": "xgb_sv", "SVR": "svr_sv"}[model_choice]
        ev_key = {"RandomForest": "rf_ev", "XGBoost": "xgb_ev", "SVR": "svr_ev"}[model_choice]
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
        st.pyplot(fig)
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
        st.pyplot(fig2)
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

    if not _os.path.exists(_csv):
        st.warning(
            "No comparison data found. "
            "Run `python src/paper_comparison.py` first."
        )
    else:
        import pandas as _pd

        df = _pd.read_csv(_csv, index_col=0)

        # ── Direction flags — lower is better except R² ────────────────────────
        higher_better = {"R²": True}

        def _highlight_best_row(row):
            """Highlight the entire best-model row with high-contrast gold style."""
            if row.name == _best_model:
                return ["background-color: #b8860b; color: #ffffff; font-weight: bold"] * len(row)
            return [""] * len(row)

        def _highlight_best_cell(col):
            """Underline the best value in every metric column."""
            hb   = higher_better.get(col.name, False)
            best = col.max() if hb else col.min()
            return [
                "border-bottom: 2px solid #ffd700" if abs(v - best) < 1e-9 else ""
                for v in col
            ]

        # Format: R² to 4 dp, others to 2 dp
        fmt = {c: ("{:.4f}" if c == "R²" else "{:.2f}") for c in df.columns}

        styled = (
            df.style
            .apply(_highlight_best_row, axis=1)
            .apply(_highlight_best_cell, axis=0)
            .format(fmt)
        )
        st.dataframe(styled, width="stretch")

        # ── Per-metric winner cards (same as Overview) ──────────────────────
        st.markdown("##### 🏆 Best model per metric")
        _winner_cards(df)

        st.markdown("---")

        # ── Grouped bar chart ─────────────────────────────────────────────
        if _os.path.exists(_png):
            st.image(_png, width="stretch")

        # ── LaTeX source ─────────────────────────────────────────────────
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