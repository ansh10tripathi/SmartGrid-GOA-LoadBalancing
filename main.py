"""
main.py
-------
End-to-end pipeline: Preprocess -> Forecast -> GOA -> Evaluate -> Visualise
Usage:  python main.py
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing      import preprocess
from src.forecasting_model  import run_forecasting_pipeline, compare_models, save_model
from src.lstm_model         import run_lstm_pipeline, WINDOW_SIZE
from src.quantile_model     import run_quantile_pipeline
from src.explainability     import run_explainability_pipeline
from src.goa_optimization   import grasshopper_optimization, plot_constraint_comparison
from src.evaluation         import compare_before_after, compute_metrics
from src.leakage_audit      import run_audit
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _save(path: str):
    """Save current figure and open it on Windows."""
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    if os.name == "nt":
        os.startfile(path)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Preprocess ───────────────────────────────────────────────────
    print("\nStep 1: Preprocessing...")
    _root = os.path.dirname(os.path.abspath(__file__))
    X_train, X_test, y_train, y_test, scaler, target_scaler, train_df, test_df = \
        preprocess(os.path.join(_root, "dataset", "DUQ_hourly.csv"))
    joblib.dump(scaler,        os.path.join(_root, "models", "minmax_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(_root, "models", "target_scaler.pkl"))
    print("  MinMaxScaler saved -> models/minmax_scaler.pkl")
    print("  TargetScaler saved -> models/target_scaler.pkl")

    # ── Step 1.1: temp_C vs Load correlation plot ────────────────────────────
    print("\nStep 1.1: Plotting temp_C vs Load correlation...")
    full_df = pd.concat([train_df, test_df])
    corr    = full_df["temp_C"].corr(full_df["load"])
    print(f"  Pearson r(temp_C, load) = {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter: temp_C vs load (sample 5k points to keep plot readable)
    sample = full_df.sample(n=min(5000, len(full_df)), random_state=42)
    axes[0].scatter(sample["temp_C"], sample["load"],
                    alpha=0.15, s=4, color="steelblue")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Load (MW)")
    axes[0].set_title(f"temp_C vs Load  (Pearson r = {corr:.3f})")
    axes[0].grid(alpha=0.3)

    # Monthly mean temp and load to show co-movement
    monthly = full_df.copy()
    monthly["month"] = pd.to_datetime(full_df["datetime"]).dt.month
    monthly_agg = monthly.groupby("month")[["temp_C", "load"]].mean()
    ax2 = axes[1]
    ax3 = ax2.twinx()
    ax2.plot(monthly_agg.index, monthly_agg["temp_C"],
             color="tomato",   marker="o", label="Mean temp_C")
    ax3.plot(monthly_agg.index, monthly_agg["load"],
             color="steelblue", marker="s", label="Mean Load (MW)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Temperature (°C)", color="tomato")
    ax3.set_ylabel("Load (MW)",        color="steelblue")
    ax2.set_title("Monthly Mean Temperature vs Load")
    ax2.set_xticks(range(1, 13))
    lines  = ax2.get_lines() + ax3.get_lines()
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Exogenous Feature: temp_C Correlation with Load", fontsize=11)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "temp_load_correlation.png")))

    # ── Step 1.2: TOU tier vs average load bar chart ──────────────────────────
    print("\nStep 1.2: Plotting TOU tier vs average load...")
    from src.preprocessing import TOU_LABELS, TOU_PRICES
    full_df  = pd.concat([train_df, test_df])
    tou_agg  = (
        full_df.groupby("tou_tier")["load"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    tier_labels = [TOU_LABELS[t] for t in tou_agg["tou_tier"]]
    tier_prices = [TOU_PRICES[t] for t in tou_agg["tou_tier"]]
    colors      = ["steelblue", "orange", "tomato"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: mean load per tier with std error bars
    bars = axes[0].bar(tier_labels, tou_agg["mean"],
                       yerr=tou_agg["std"], capsize=5,
                       color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, tou_agg["mean"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + tou_agg["std"].max() * 0.05,
                     f"{val:.0f} MW", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Average Load by TOU Tier")
    axes[0].set_ylabel("Mean Load (MW)")
    axes[0].set_xlabel("TOU Tier")
    axes[0].grid(axis="y", alpha=0.3)

    # Right: hourly mean load coloured by tier
    hourly_load = full_df.groupby(
        full_df["datetime"].dt.hour
    )["load"].mean()
    hour_tiers  = [int(full_df[full_df["datetime"].dt.hour == h]["tou_tier"].iloc[0])
                   for h in hourly_load.index]
    bar_colors  = [colors[t] for t in hour_tiers]
    axes[1].bar(hourly_load.index, hourly_load.values,
                color=bar_colors, alpha=0.85, width=0.8)
    axes[1].set_title("Hourly Mean Load (coloured by TOU tier)")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Mean Load (MW)")
    axes[1].set_xticks(range(0, 24, 2))
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[t], label=f"{TOU_LABELS[t]}  ${TOU_PRICES[t]:.2f}")
                       for t in range(3)]
    axes[1].legend(handles=legend_elements, fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("TOU Tier Validation: Pricing Signal vs Observed Load", fontsize=11)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "tou_tier_validation.png")))

    # ── Step 2: Train RF + SVR + XGBoost, compare, pick best ────────────────
    print("\nStep 2: Training & Comparing Models (RF, SVR, XGBoost)...")
    from src.forecasting_model import train_random_forest, train_svr, train_xgboost, evaluate_model
    from src.evaluation import evaluate_model_performance
    rf_pipeline  = train_random_forest(X_train, y_train, use_search=True)
    svr_pipeline = train_svr(X_train,            y_train, use_search=True)
    xgb_pipeline = train_xgboost(X_train,        y_train, use_search=True)

    rf_metrics,  y_pred_rf  = evaluate_model(rf_pipeline,  X_test, y_test, target_scaler)
    svr_metrics, y_pred_svr = evaluate_model(svr_pipeline, X_test, y_test, target_scaler)
    xgb_metrics, y_pred_xgb = evaluate_model(xgb_pipeline, X_test, y_test, target_scaler)

    all_metrics = {"RandomForest": rf_metrics, "SVR": svr_metrics, "XGBoost": xgb_metrics}
    all_preds   = {"RandomForest": y_pred_rf,  "SVR": y_pred_svr,  "XGBoost": y_pred_xgb}

    # Select best model by R2 (should be XGBoost after fixing scaling)
    candidates = [
        ("RandomForest", rf_pipeline,  rf_metrics,  y_pred_rf),
        ("SVR",          svr_pipeline, svr_metrics, y_pred_svr),
        ("XGBoost",      xgb_pipeline, xgb_metrics, y_pred_xgb),
    ]
    best_name, model, metrics, y_pred = max(candidates, key=lambda t: t[2]["R2"])
    print(f"  Best model: {best_name}  R²={metrics['R2']:.4f}")
    save_model(model)
    
    # Inverse-transform y_pred for visualization (it's currently scaled)
    from src.evaluation import inverse_transform_predictions
    y_pred_mw = inverse_transform_predictions(np.asarray(y_pred), target_scaler)
    y_test_mw = inverse_transform_predictions(np.asarray(y_test), target_scaler)
    evaluate_model_performance(y_test_mw, y_pred_mw, target_scaler=None)  # Already in MW

    # ── Save individual model artefacts for paper_comparison / evaluation ────
    # evaluation._load_sklearn_models() looks for these specific filenames.
    joblib.dump(rf_pipeline,  os.path.join(_root, "models", "random_forest_model.pkl"))
    joblib.dump(svr_pipeline, os.path.join(_root, "models", "svr_model.pkl"))
    joblib.dump(xgb_pipeline, os.path.join(_root, "models", "xgboost_model.pkl"))
    print("  Individual model artefacts saved -> models/")

    # ── Step 2.1: Model comparison bar chart (R²) ────────────────────────────
    print("\nStep 2.1: Model Comparison Bar Chart...")

    # Step 2.5: LSTM
    print("\nStep 2.5: Training LSTM...")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    # CRITICAL: Pass target_scaler to LSTM for consistent scaling across all models
    lstm_metrics, lstm_pred, lstm_true, lstm_tl, lstm_vl = run_lstm_pipeline(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        target_scaler=target_scaler,  # SHARED scaler for fair comparison
        device=device,
    )
    print(f"  LSTM metrics: {lstm_metrics}")
    print(f"  LSTM predictions: min={lstm_pred.min():.2f} MW, max={lstm_pred.max():.2f} MW")
    print(f"  LSTM ground truth: min={lstm_true.min():.2f} MW, max={lstm_true.max():.2f} MW")
    all_metrics["LSTM"] = lstm_metrics

    # LSTM training curve
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(lstm_tl, label="Train loss", color="steelblue")
    ax.plot(lstm_vl, label="Val loss",   color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalised)")
    ax.set_title("LSTM Training Curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "lstm_training_curve.png")))

    # LSTM vs XGBoost vs actual (first 336 h = 2 weeks)
    n_plot     = 336
    # y_pred_mw is in MW, need to align with LSTM window offset
    ml_aligned = y_pred_mw[WINDOW_SIZE : WINDOW_SIZE + n_plot]
    lstm_plot  = lstm_pred[:n_plot]
    true_plot  = lstm_true[:n_plot]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(true_plot,  label="Actual",
            color="black",      linewidth=1.2)
    ax.plot(ml_aligned, label=f"XGBoost (R2={all_metrics['XGBoost']['R2']:.4f})",
            color="seagreen",   linewidth=1.0, linestyle="--")
    ax.plot(lstm_plot,  label=f"LSTM    (R2={lstm_metrics['R2']:.4f})",
            color="darkorange", linewidth=1.0)
    ax.set_title("LSTM vs XGBoost vs Actual Load (first 2 weeks of test set)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Load (MW)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "lstm_vs_ml_comparison.png")))

    # model_comparison.png — built after ALL models (RF/SVR/XGB/LSTM/QR) are ready
    # deferred to after quantile step below

    # ── Step 2.2: Feature Importance (RF only) ───────────────────────────────
    from src.preprocessing import FEATURE_COLS
    print("\nStep 2.2: Feature Importance (RandomForest)...")
    imp_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": rf_pipeline["model"].feature_importances_,
    }).sort_values(by="Importance", ascending=False)
    print(imp_df)
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["Feature"], imp_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (RandomForest)")
    plt.xlabel("Importance")
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "feature_importance.png")))

    # ── Step 2.6: Quantile regression ────────────────────────────────────────
    print("\nStep 2.6: Training Quantile Regression (q=0.10/0.50/0.90)...")
    q_metrics = run_quantile_pipeline(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
        target_scaler=target_scaler,
    )
    print(f"  Quantile metrics: {q_metrics}")
    all_metrics["QR-Median"] = {
        "RMSE": q_metrics["RMSE"],
        "MAE":  q_metrics["MAE"],
        "R2":   q_metrics["R2"],
    }

    # ── Step 2.1 (deferred): model_comparison.png — all 5 models ─────────────
    print("\nStep 2.1: Model Comparison Bar Chart (all models)...")
    _BAR_COLORS = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]
    model_names = list(all_metrics.keys())          # RF, SVR, XGB, LSTM, QR-Median
    r2_values   = [all_metrics[m]["R2"]   for m in model_names]
    rmse_values = [all_metrics[m]["RMSE"] for m in model_names]
    bar_colors  = _BAR_COLORS[:len(model_names)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bars = axes[0].bar(model_names, r2_values, color=bar_colors, alpha=0.85, width=0.5)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                     f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Model Comparison — R²")
    axes[0].set_ylabel("R²")
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(model_names, rmse_values, color=bar_colors, alpha=0.85, width=0.5)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                     f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Model Comparison — RMSE (MW)")
    axes[1].set_ylabel("RMSE (MW)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("ML Model Comparison (all models)", fontsize=12)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "model_comparison.png")))

    # ── Step 2.7: SHAP explainability ────────────────────────────────────────
    print("\nStep 2.7: Computing SHAP values (RF + XGBoost + SVR)...")
    run_explainability_pipeline(
        rf_pipeline, svr_pipeline, xgb_pipeline,
        X_train.values, X_test.values, y_test.values,
        FEATURE_COLS,
    )

    # ── Step 2.8: Paper comparison table (uses saved model artefacts) ─────────
    print("\nStep 2.8: Building publication comparison table...")
    from src.paper_comparison import run_paper_comparison
    run_paper_comparison(X_test=X_test, y_test=y_test, target_scaler=target_scaler)

    # ── Step 3: Leakage audit ───────────────────────────────────────────────
    print("\nStep 3: Running Leakage Audit...")
    run_audit(model, X_test, y_test, train_df, test_df, target_scaler=target_scaler)

    # ── Step 4: GOA optimisation (best model predictions) ────────────────────
    print("\nStep 4: Running GOA Optimization (best model)...")
    # y_pred_mw is already in MW units from the inverse transform above
    
    # CRITICAL VALIDATION: Ensure predictions match test set length
    assert len(y_pred_mw) == len(X_test), (
        f"LEAKAGE DETECTED: y_pred_mw length ({len(y_pred_mw)}) != "
        f"X_test length ({len(X_test)}). Predictions must be on test set only."
    )
    print(f"  [GOA] Validation passed: {len(y_pred_mw)} predictions == {len(X_test)} test samples")
    
    price_test = test_df["tou_price"].values[:len(y_pred_mw)]

    goa_result     = grasshopper_optimization(
        predicted_load=y_pred_mw, price=price_test,
        n_grasshoppers=30, max_iter=100,
    )
    optimized_load = goa_result["optimized_load"]

    # ── Step 4: Evaluate before vs after ─────────────────────────────────────
    print("\nStep 4: Evaluating Results...")
    compare_before_after(y_pred_mw, optimized_load, price_test)

    # ── Step 4b: Constraint comparison plot ──────────────────────────────────
    print("\nStep 4b: Plotting constraint comparison...")
    plot_constraint_comparison(
        y_pred_mw, optimized_load,
        goa_result["max_ramp_rate"],
        goa_result["grid_max"],
        goa_result["load_min"],
        save_path=os.path.join(RESULTS_DIR, "constraint_comparison.png"),
    )

    # ── Step 5: Visualisations ────────────────────────────────────────────────

    # 5a – Before vs After load curve
    plt.figure(figsize=(12, 5))
    plt.plot(y_pred_mw,        label="Before GOA", color="tomato",   linewidth=1.4)
    plt.plot(optimized_load, label="After GOA",  color="seagreen", linewidth=1.4)
    plt.axhline(y_pred_mw.max(),        color="red",       linestyle=":",  linewidth=1,
                label=f"Peak before = {y_pred_mw.max():.1f} MW")
    plt.axhline(optimized_load.max(), color="darkgreen", linestyle=":",  linewidth=1,
                label=f"Peak after  = {optimized_load.max():.1f} MW")
    plt.title("Load Schedule: Before vs After GOA Optimisation")
    plt.xlabel("Time Step (hours)")
    plt.ylabel("Load (MW)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "before_after_load.png")))

    # 🔥 Extra Zoomed Comparison (first 200 points)
    plt.figure(figsize=(10, 4))
    plt.plot(y_pred_mw[:200], label="Before GOA", color="tomato")
    plt.plot(optimized_load[:200], label="After GOA", color="seagreen")
    plt.legend()
    plt.title("GOA Load Comparison (First 200 Points)")
    plt.xlabel("Time Step")
    plt.ylabel("Load (MW)")
    plt.grid(alpha=0.3)

    _save(os.path.abspath(os.path.join(RESULTS_DIR, "goa_comparison.png")))

    # 5b – Cost comparison bar chart
    before_cost = float(np.sum(y_pred_mw * price_test))
    after_cost  = float(np.sum(optimized_load * price_test))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Before GOA", "After GOA"], [before_cost, after_cost],
                   color=["tomato", "seagreen"], alpha=0.85, width=0.4)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"${bar.get_height():,.1f}", ha="center", va="bottom", fontsize=9)
    plt.title("Total Electricity Cost: Before vs After GOA")
    plt.ylabel("Total Cost ($)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "cost_comparison.png")))

    # 5c – GOA convergence curve
    plt.figure(figsize=(8, 4))
    plt.plot(goa_result["fitness_history"], color="purple", linewidth=1.5)
    plt.title("GOA Convergence - Best Fitness over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "goa_convergence.png")))

    # 5d – Normalised performance comparison (separate subplots — different scales)
    m_before = compute_metrics(y_pred_mw,         price_test, "Before GOA")
    m_after  = compute_metrics(optimized_load, price_test, "After GOA")
    kpi_keys   = ["peak_load", "total_cost", "PAR", "variance"]
    kpi_labels = ["Peak Load (MW)", "Total Cost ($)", "PAR", "Variance"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, key, label in zip(axes, kpi_keys, kpi_labels):
        vals  = [m_before[key], m_after[key]]
        bars  = ax.bar(["Before", "After"], vals,
                       color=["tomato", "seagreen"], alpha=0.85, width=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        pct = (vals[1] - vals[0]) / vals[0] * 100
        ax.set_xlabel(f"{pct:+.1f}%", fontsize=8,
                      color="seagreen" if pct < 0 else "tomato")
    fig.suptitle("Performance Comparison: Before vs After GOA", fontsize=11)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "performance_comparison.png")))

    print("\nProject Execution Completed! Results saved to results/")

    # ── Step 5e: Algorithm comparison CSV + PNG (GOA KPIs table) ─────────────
    print("\nStep 5e: Saving algorithm comparison table...")
    m_before = compute_metrics(y_pred_mw,         price_test, "Before GOA")
    m_after  = compute_metrics(optimized_load, price_test, "After GOA")
    algo_df  = pd.DataFrame([
        {
            "Metric":     "Peak Load (MW)",
            "Before GOA": round(m_before["peak_load"],  2),
            "After GOA":  round(m_after["peak_load"],   2),
            "Change (%)": round((m_after["peak_load"]  - m_before["peak_load"])  / m_before["peak_load"]  * 100, 2),
        },
        {
            "Metric":     "Total Cost ($)",
            "Before GOA": round(m_before["total_cost"], 2),
            "After GOA":  round(m_after["total_cost"],  2),
            "Change (%)": round((m_after["total_cost"] - m_before["total_cost"]) / m_before["total_cost"] * 100, 2),
        },
        {
            "Metric":     "PAR",
            "Before GOA": round(m_before["PAR"],        4),
            "After GOA":  round(m_after["PAR"],         4),
            "Change (%)": round((m_after["PAR"]         - m_before["PAR"])         / m_before["PAR"]         * 100, 2),
        },
        {
            "Metric":     "Variance (MW²)",
            "Before GOA": round(m_before["variance"],   2),
            "After GOA":  round(m_after["variance"],    2),
            "Change (%)": round((m_after["variance"]    - m_before["variance"])    / m_before["variance"]    * 100, 2),
        },
    ])
    algo_csv = os.path.join(RESULTS_DIR, "algorithm_comparison.csv")
    algo_df.to_csv(algo_csv, index=False)
    print(f"  algorithm_comparison.csv saved -> {algo_csv}")

    # Grouped bar chart: Before vs After for each KPI
    kpi_labels = algo_df["Metric"].tolist()
    before_vals = algo_df["Before GOA"].tolist()
    after_vals  = algo_df["After GOA"].tolist()
    x = np.arange(len(kpi_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - w/2, before_vals, w, label="Before GOA", color="tomato",   alpha=0.85)
    b2 = ax.bar(x + w/2, after_vals,  w, label="After GOA",  color="seagreen", alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(kpi_labels, fontsize=9)
    ax.set_title("Algorithm Comparison: Before vs After GOA", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "algorithm_comparison.png")))

    # ── Save unified model_results.json (single source of truth for app.py) ──
    import json
    from src.evaluation import _mape

    def _safe_float(v):
        """Convert numpy scalars to plain Python float."""
        try:
            return float(v)
        except Exception:
            return None

    # GOA KPIs
    goa_kpis = {
        "peak_before":    _safe_float(y_pred_mw.max()),
        "peak_after":     _safe_float(optimized_load.max()),
        "peak_pct":       round((_safe_float(optimized_load.max()) - _safe_float(y_pred_mw.max())) / _safe_float(y_pred_mw.max()) * 100, 2),
        "cost_before":    _safe_float(np.sum(y_pred_mw * price_test)),
        "cost_after":     _safe_float(np.sum(optimized_load * price_test)),
        "cost_pct":       round((_safe_float(np.sum(optimized_load * price_test)) - _safe_float(np.sum(y_pred_mw * price_test))) / _safe_float(np.sum(y_pred_mw * price_test)) * 100, 2),
        "par_before":     _safe_float(float(y_pred_mw.max()) / float(y_pred_mw.mean())),
        "par_after":      _safe_float(float(optimized_load.max()) / float(optimized_load.mean())),
        "var_before":     _safe_float(np.var(y_pred_mw)),
        "var_after":      _safe_float(np.var(optimized_load)),
        "best_model":     best_name,
        "best_fitness":   _safe_float(goa_result["best_fitness"]),
        "max_ramp_rate":  _safe_float(goa_result["max_ramp_rate"]),
        "grid_max":       _safe_float(goa_result["grid_max"]),
        "load_min":       _safe_float(goa_result["load_min"]),
        "ramp_violations":   goa_result["constraints"]["ramp_violations"],
        "cap_violations":    goa_result["constraints"]["cap_violations"],
        "floor_violations":  goa_result["constraints"]["floor_violations"],
        "feasible":          goa_result["constraints"]["feasible"],
    }

    # ML metrics — normalise key names to RMSE/MAE/R2/MAPE everywhere
    def _norm(m: dict) -> dict:
        """Ensure every metrics dict has RMSE, MAE, R2, MAPE keys."""
        out = {}
        out["RMSE"] = _safe_float(m.get("RMSE") or m.get("RMSE (MW)"))
        out["MAE"]  = _safe_float(m.get("MAE")  or m.get("MAE (MW)"))
        out["R2"]   = _safe_float(m.get("R2")   or m.get("R²"))
        out["MAPE"] = _safe_float(m.get("MAPE") or m.get("MAPE (%)"))
        return out

    y_test_arr = np.asarray(y_test)
    from src.evaluation import inverse_transform_predictions
    y_test_mw = inverse_transform_predictions(y_test_arr, target_scaler)

    model_results = {
        "RandomForest": {
            **_norm(rf_metrics),
            "MAPE": round(_mape(
                y_test_mw,
                inverse_transform_predictions(np.asarray(y_pred_rf), target_scaler)
            ), 4),
        },
        "SVR": {
            **_norm(svr_metrics),
            "MAPE": round(_mape(
                y_test_mw,
                inverse_transform_predictions(np.asarray(y_pred_svr), target_scaler)
            ), 4),
        },
        "XGBoost": {
            **_norm(xgb_metrics),
            "MAPE": round(_mape(
                y_test_mw,
                inverse_transform_predictions(np.asarray(y_pred_xgb), target_scaler)
            ), 4),
        },
        "LSTM": {
            **_norm(lstm_metrics),
            "MAPE": round(_mape(np.asarray(lstm_true), np.asarray(lstm_pred)), 4),
        },
        "QuantileGBR": {
            "RMSE":     _safe_float(q_metrics.get("RMSE")),
            "MAE":      _safe_float(q_metrics.get("MAE")),
            "R2":       _safe_float(q_metrics.get("R2")),
            "MAPE":     None,
            "coverage": _safe_float(q_metrics.get("coverage_80pct")),
            "width_MW": _safe_float(q_metrics.get("mean_interval_MW")),
        },
        "_meta": {
            "best_model":   best_name,
            "train_rows":   int(len(X_train)),
            "test_rows":    int(len(X_test)),
            "n_features":   int(X_train.shape[1]),
            "split_date":   str(train_df["datetime"].max()),
            "goa":          goa_kpis,
        },
    }

    results_json = os.path.join(RESULTS_DIR, "model_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2)
    print(f"  model_results.json saved -> {results_json}")


if __name__ == "__main__":
    main()

