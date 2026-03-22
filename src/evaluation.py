"""
src/evaluation.py
-----------------
Two evaluation layers:

  1. ML Model Performance  – RMSE, MAE, R²  (evaluate_model_performance)
  2. Grid KPIs             – Peak Load, Cost, PAR, Variance (compare_before_after)

Metric reference
----------------
  RMSE  – Root Mean Square Error: average prediction error in the same unit as
          load (kWh). Penalises large errors more than MAE. Lower is better.

  MAE   – Mean Absolute Error: average absolute deviation between actual and
          predicted load (kWh). Easier to interpret than RMSE. Lower is better.

  R²    – Coefficient of Determination: proportion of variance in the target
          explained by the model. Range [0, 1]; closer to 1 is better.

  Peak Load  – maximum load in the schedule (kWh)
  Total Cost – sum(load * price)  ($)
  PAR        – Peak-to-Average Ratio  = peak / mean
  Variance   – statistical variance of the load schedule
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # file-safe backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RESULTS_DIR = "results"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 – ML Model Performance
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model_performance(
    y_test,
    y_pred,
    save_plot: bool = True,
    plot_filename: str = "actual_vs_predicted.png",
    n_display: int = 100,
) -> dict:
    """
    Calculate RMSE, MAE, and R² for a regression model and plot
    Actual vs Predicted values.

    Parameters
    ----------
    y_test        : array-like – ground-truth load values
    y_pred        : array-like – model-predicted load values
    save_plot     : save the plot to results/ when True
    plot_filename : output filename inside results/
    n_display     : number of samples shown in the plot

    Returns
    -------
    dict with keys: RMSE, MAE, R2
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # ── Compute metrics ───────────────────────────────────────────────────────
    # RMSE: same unit as load (kWh); heavily penalises large errors
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAE: average absolute error in kWh; robust to outliers
    mae  = mean_absolute_error(y_test, y_pred)

    # R²: fraction of variance explained; 1.0 = perfect, 0.0 = baseline mean
    r2   = r2_score(y_test, y_pred)

    metrics = {
        "RMSE": round(rmse, 4),
        "MAE":  round(mae,  4),
        "R2":   round(r2,   4),
    }

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("        REGRESSION MODEL PERFORMANCE METRICS")
    print("=" * 55)
    print(f"  RMSE  (Root Mean Square Error) : {rmse:>10.4f} kWh")
    print(f"        → avg error penalising large deviations")
    print(f"  MAE   (Mean Absolute Error)    : {mae:>10.4f} kWh")
    print(f"        → avg absolute prediction error")
    print(f"  R²    (Coefficient of Det.)    : {r2:>10.4f}")
    print(f"        → {r2*100:.1f}% of load variance explained by model")
    print("=" * 55)

    # ── Plot Actual vs Predicted ──────────────────────────────────────────────
    n   = min(n_display, len(y_test))
    idx = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [3, 1]})

    # Top panel – line plot
    axes[0].plot(idx, y_test[:n], label="Actual Load",
                 color="steelblue", linewidth=1.6)
    axes[0].plot(idx, y_pred[:n], label="Predicted Load",
                 color="tomato",    linewidth=1.6, linestyle="--")
    axes[0].set_title(
        f"Actual vs Predicted Energy Load  "
        f"(RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.4f})",
        fontsize=11,
    )
    axes[0].set_ylabel("Load (kWh)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # Bottom panel – residual bar chart
    residuals = y_test[:n] - y_pred[:n]
    colors    = ["seagreen" if r >= 0 else "tomato" for r in residuals]
    axes[1].bar(idx, residuals, color=colors, alpha=0.7, width=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Residual (kWh)")
    axes[1].set_title("Residuals (Actual − Predicted)", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, plot_filename)
        plt.savefig(out_path, dpi=150)
        print(f"[evaluate_model_performance] Plot saved → {out_path}")

    plt.show()
    return metrics


# ── 1. Individual metrics ─────────────────────────────────────────────────────

def peak_load(schedule: np.ndarray) -> float:
    return float(np.max(schedule))


def total_cost(schedule: np.ndarray, price: np.ndarray) -> float:
    return float(np.sum(schedule * price))


def par(schedule: np.ndarray) -> float:
    """Peak-to-Average Ratio."""
    mean = np.mean(schedule)
    return float(np.max(schedule) / mean) if mean != 0 else 0.0


def variance(schedule: np.ndarray) -> float:
    return float(np.var(schedule))


# ── 2. Full metrics dict ──────────────────────────────────────────────────────

def compute_metrics(schedule: np.ndarray, price: np.ndarray, label: str = "") -> dict:
    """
    Return a dict of all KPIs for a given load schedule.
    """
    metrics = {
        "label":      label,
        "peak_load":  round(peak_load(schedule),  4),
        "total_cost": round(total_cost(schedule, price), 4),
        "PAR":        round(par(schedule),         4),
        "variance":   round(variance(schedule),    4),
        "mean_load":  round(float(np.mean(schedule)), 4),
    }
    return metrics


# ── 3. Comparison ─────────────────────────────────────────────────────────────

def compare_before_after(
    before_schedule: np.ndarray,
    after_schedule:  np.ndarray,
    price:           np.ndarray,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame and print a summary.
    Returns a DataFrame with one row per metric.
    """
    before = compute_metrics(before_schedule, price, label="Before GOA")
    after  = compute_metrics(after_schedule,  price, label="After GOA")

    rows = []
    for key in ["peak_load", "total_cost", "PAR", "variance", "mean_load"]:
        b_val = before[key]
        a_val = after[key]
        change_pct = ((a_val - b_val) / b_val * 100) if b_val != 0 else 0.0
        rows.append({
            "Metric":        key,
            "Before GOA":    b_val,
            "After GOA":     a_val,
            "Change (%)":    round(change_pct, 2),
            "Improved":      "✅" if a_val < b_val else "❌",
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("         BEFORE vs AFTER GOA OPTIMISATION")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    return df


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    n = 100

    # Simulate model predictions
    y_test_demo = np.random.uniform(150, 380, n)
    y_pred_demo = y_test_demo + np.random.normal(0, 18, n)   # add noise

    # Layer 1 – ML metrics
    ml_metrics = evaluate_model_performance(y_test_demo, y_pred_demo)
    print("\nReturned metrics dict:", ml_metrics)

    # Layer 2 – Grid KPIs
    price = np.random.uniform(0.08, 0.15, n)
    after = y_pred_demo * np.random.uniform(0.85, 0.98, n)
    df    = compare_before_after(y_pred_demo, after, price)
    print("\nKPI DataFrame:\n", df)
