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

from src.preprocessing     import preprocess
from src.forecasting_model import run_forecasting_pipeline
from src.goa_optimization  import grasshopper_optimization
from src.evaluation        import compare_before_after, compute_metrics
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
    X, y, scaler, raw_df = preprocess(os.path.join(_root, "dataset", "DUQ_hourly.csv"))
    joblib.dump(scaler, os.path.join(_root, "models", "minmax_scaler.pkl"))
    print("  MinMaxScaler saved -> models/minmax_scaler.pkl")

    # ── Step 2: Train RF + SVR + XGBoost, compare, pick best ────────────────
    print("\nStep 2: Training & Comparing Models (RF, SVR, XGBoost)...")
    model, metrics, y_test, y_pred, X_test, all_metrics, all_preds = \
        run_forecasting_pipeline(X, y, use_search=True)
    print("  Best model metrics:", metrics)

    # ── Step 2.1: Model comparison bar chart (R²) ────────────────────────────
    print("\nStep 2.1: Model Comparison Bar Chart...")
    model_names = list(all_metrics.keys())
    r2_values   = [all_metrics[m]["R2"]   for m in model_names]
    rmse_values = [all_metrics[m]["RMSE"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["steelblue", "tomato", "seagreen"]

    bars = axes[0].bar(model_names, r2_values, color=colors, alpha=0.85, width=0.4)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                     f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Model Comparison — R²")
    axes[0].set_ylabel("R²")
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(model_names, rmse_values, color=colors, alpha=0.85, width=0.4)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Model Comparison — RMSE")
    axes[1].set_ylabel("RMSE (kWh)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("ML Model Comparison", fontsize=12)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "model_comparison.png")))

    # ── Step 2.2: Feature Importance (RF only) ───────────────────────────────
    from src.preprocessing import FEATURE_COLS
    # model may be SVR/XGB bundle; feature importance only for RF
    if hasattr(model, "feature_importances_"):
        print("\nStep 2.2: Feature Importance...")
        imp_df = pd.DataFrame({
            "Feature": FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        print(imp_df)
        plt.figure(figsize=(8, 5))
        plt.barh(imp_df["Feature"], imp_df["Importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance (Best Model)")
        plt.xlabel("Importance")
        _save(os.path.abspath(os.path.join(RESULTS_DIR, "feature_importance.png")))

    # ── Step 3: GOA optimisation (best model predictions) ────────────────────
    print("\nStep 3: Running GOA Optimization (best model)...")
    test_start  = len(y) - len(y_test)
    price_test  = raw_df["tou_price"].values[test_start : test_start + len(y_test)]

    goa_result     = grasshopper_optimization(
        predicted_load=y_pred, price=price_test,
        n_grasshoppers=30, max_iter=100,
    )
    optimized_load = goa_result["optimized_load"]

    # ── Step 4: Evaluate before vs after ─────────────────────────────────────
    print("\nStep 4: Evaluating Results...")
    compare_before_after(y_pred, optimized_load, price_test)

    # ── Step 5: Visualisations ────────────────────────────────────────────────

    # 5a – Before vs After load curve
    plt.figure(figsize=(12, 5))
    plt.plot(y_pred,        label="Before GOA", color="tomato",   linewidth=1.4)
    plt.plot(optimized_load, label="After GOA",  color="seagreen", linewidth=1.4)
    plt.axhline(y_pred.max(),        color="red",       linestyle=":",  linewidth=1,
                label=f"Peak before = {y_pred.max():.1f} kWh")
    plt.axhline(optimized_load.max(), color="darkgreen", linestyle=":",  linewidth=1,
                label=f"Peak after  = {optimized_load.max():.1f} kWh")
    plt.title("Load Schedule: Before vs After GOA Optimisation")
    plt.xlabel("Time Step (hours)")
    plt.ylabel("Load (kWh)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(os.path.abspath(os.path.join(RESULTS_DIR, "before_after_load.png")))

    # 🔥 Extra Zoomed Comparison (first 200 points)
    plt.figure(figsize=(10, 4))
    plt.plot(y_pred[:200], label="Before GOA", color="tomato")
    plt.plot(optimized_load[:200], label="After GOA", color="seagreen")
    plt.legend()
    plt.title("GOA Load Comparison (First 200 Points)")
    plt.xlabel("Time Step")
    plt.ylabel("Load (kWh)")
    plt.grid(alpha=0.3)

    _save(os.path.abspath(os.path.join(RESULTS_DIR, "goa_comparison.png")))

    # 5b – Cost comparison bar chart
    before_cost = float(np.sum(y_pred * price_test))
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
    m_before = compute_metrics(y_pred,         price_test, "Before GOA")
    m_after  = compute_metrics(optimized_load, price_test, "After GOA")
    kpi_keys   = ["peak_load", "total_cost", "PAR", "variance"]
    kpi_labels = ["Peak Load (kWh)", "Total Cost ($)", "PAR", "Variance"]

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


if __name__ == "__main__":
    main()

