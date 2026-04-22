"""
src/evaluation.py
-----------------
Three evaluation layers:

  1. ML Model Performance  - RMSE, MAE, R², MAPE  (evaluate_model_performance)
  2. Grid KPIs             - Peak Load, Cost, PAR, Variance (compare_before_after)
  3. Publication Table     - build_model_comparison_table
                             Loads every saved model from models/, runs inference
                             on the supplied test set, computes RMSE / MAE / R² /
                             MAPE, renders a booktabs LaTeX table with bold best
                             per metric, and saves a grouped bar chart.
                             No re-training — uses the serialised artefacts only.

Metric reference
----------------
  RMSE  - Root Mean Square Error: average prediction error in MW.
          Penalises large errors more than MAE. Lower is better.

  MAE   - Mean Absolute Error: average absolute deviation in MW.
          Easier to interpret than RMSE. Lower is better.

  R²    - Coefficient of Determination: proportion of variance explained.
          Range [0, 1]; closer to 1 is better.

  MAPE  - Mean Absolute Percentage Error (%): scale-free relative error.
          Lower is better.

  Peak Load  – maximum load in the schedule (MW)
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
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SRC_DIR)
RESULTS_DIR = os.path.join(_ROOT, "results")
MODELS_DIR  = os.path.join(_ROOT, "models")


# ── Shared metric helpers ─────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE excluding near-zero actuals to avoid division by zero."""
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def inverse_transform_predictions(y_scaled: np.ndarray, target_scaler) -> np.ndarray:
    """Inverse-transform a 1-D scaled array back to original MW units."""
    return target_scaler.inverse_transform(
        np.asarray(y_scaled).reshape(-1, 1)
    ).ravel()


def _compute_four_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, R², MAPE rounded to 4 d.p. (values must be on the same scale)."""
    return {
        "RMSE (MW)": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE (MW)":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "R²":        round(float(r2_score(y_true, y_pred)), 4),
        "MAPE (%)":  round(_mape(y_true, y_pred), 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1 — ML Model Performance
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model_performance(
    y_test,
    y_pred,
    target_scaler=None,
    save_plot: bool = True,
    plot_filename: str = "actual_vs_predicted.png",
    n_display: int = 100,
) -> dict:
    """
    Calculate RMSE, MAE, R², MAPE for a regression model and plot
    Actual vs Predicted values with a residual panel.

    Parameters
    ----------
    y_test        : array-like – ground-truth load values
    y_pred        : array-like – model-predicted load values
    save_plot     : save the plot to results/ when True
    plot_filename : output filename inside results/
    n_display     : number of samples shown in the plot

    Returns
    -------
    dict with keys: RMSE, MAE, R2, MAPE
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    # Inverse-transform to MW if scaler provided; otherwise assume already in MW
    if target_scaler is not None:
        y_test_mw = inverse_transform_predictions(y_test, target_scaler)
        y_pred_mw = inverse_transform_predictions(y_pred, target_scaler)
        scale_note = "(inverse-transformed to MW)"
    else:
        y_test_mw = y_test
        y_pred_mw = y_pred
        scale_note = "(values passed as-is)"

    rmse = float(np.sqrt(mean_squared_error(y_test_mw, y_pred_mw)))
    mae  = float(mean_absolute_error(y_test_mw, y_pred_mw))
    r2   = float(r2_score(y_test_mw, y_pred_mw))
    mape = _mape(y_test_mw, y_pred_mw)

    metrics = {
        "RMSE": round(rmse, 4),
        "MAE":  round(mae,  4),
        "R2":   round(r2,   4),
        "MAPE": round(mape, 4),
    }

    print("\n" + "=" * 55)
    print("        REGRESSION MODEL PERFORMANCE METRICS")
    print(f"        {scale_note}")
    print("=" * 55)
    print(f"  RMSE  (Root Mean Square Error) : {rmse:>10.4f} MW")
    print(f"        -> avg error penalising large deviations")
    print(f"  MAE   (Mean Absolute Error)    : {mae:>10.4f} MW")
    print(f"        -> avg absolute prediction error")
    print(f"  R²    (Coefficient of Det.)    : {r2:>10.4f}")
    print(f"        -> {r2*100:.1f}% of load variance explained by model")
    print(f"  MAPE  (Mean Abs. Pct. Error)   : {mape:>10.4f} %")
    print("=" * 55)

    # Swap back so the plot uses MW values
    y_test = y_test_mw
    y_pred = y_pred_mw

    # ── Plot Actual vs Predicted ──────────────────────────────────────────────
    n   = min(n_display, len(y_test))
    idx = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(idx, y_test[:n], label="Actual Load",
                 color="steelblue", linewidth=1.6)
    axes[0].plot(idx, y_pred[:n], label="Predicted Load",
                 color="tomato", linewidth=1.6, linestyle="--")
    axes[0].set_title(
        f"Actual vs Predicted Energy Load  "
        f"(RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.4f} | MAPE={mape:.2f}%)",
        fontsize=11,
    )
    axes[0].set_ylabel("Load (MW)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    residuals = y_test[:n] - y_pred[:n]
    colors    = ["seagreen" if r >= 0 else "tomato" for r in residuals]
    axes[1].bar(idx, residuals, color=colors, alpha=0.7, width=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Residual (MW)")
    axes[1].set_title("Residuals (Actual - Predicted)", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.abspath(os.path.join(RESULTS_DIR, plot_filename))
        plt.savefig(out_path, dpi=150)
        print(f"[evaluate_model_performance] Plot saved -> {out_path}")
        plt.close()
        if os.name == "nt":
            os.startfile(out_path)
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Grid KPIs
# ═════════════════════════════════════════════════════════════════════════════

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


def compute_metrics(schedule: np.ndarray, price: np.ndarray, label: str = "") -> dict:
    """Return a dict of all KPIs for a given load schedule."""
    return {
        "label":      label,
        "peak_load":  round(peak_load(schedule),        4),
        "total_cost": round(total_cost(schedule, price), 4),
        "PAR":        round(par(schedule),               4),
        "variance":   round(variance(schedule),          4),
        "mean_load":  round(float(np.mean(schedule)),    4),
    }


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
            "Metric":     key,
            "Before GOA": b_val,
            "After GOA":  a_val,
            "Change (%)": round(change_pct, 2),
            "Improved":   "YES" if a_val < b_val else "NO",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("         BEFORE vs AFTER GOA OPTIMISATION")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Publication Model Comparison Table
# ═════════════════════════════════════════════════════════════════════════════

# Metric direction and display config
_HIGHER_BETTER = {"RMSE (MW)": False, "MAE (MW)": False, "R²": True, "MAPE (%)": False}

_LATEX_HEADERS = {
    "RMSE (MW)": r"RMSE (MW)$\downarrow$",
    "MAE (MW)":  r"MAE (MW)$\downarrow$",
    "R²":        r"$R^2$$\uparrow$",
    "MAPE (%)":  r"MAPE (\%)$\downarrow$",
}

_MODEL_COLORS = {
    "Random Forest": "#4C72B0",
    "XGBoost":       "#55A868",
    "SVR":           "#DD8452",
    "LSTM":          "#C44E52",
}

_CHART_SUBTITLES = {
    "RMSE (MW)": "RMSE (MW)\n(lower = better)",
    "MAE (MW)":  "MAE (MW)\n(lower = better)",
    "R²":        "R²\n(higher = better)",
    "MAPE (%)":  "MAPE (%)\n(lower = better)",
}


def _best_per_metric(df: pd.DataFrame) -> dict:
    """Return {metric: best_value} respecting direction."""
    return {
        col: (df[col].max() if _HIGHER_BETTER.get(col, False) else df[col].min())
        for col in df.columns
    }


def _load_sklearn_models() -> dict:
    """
    Load saved sklearn Pipeline artefacts from models/.

    Returns {display_name: pipeline} for every .pkl found that matches
    a known model name.  Missing files are silently skipped so the
    function works whether or not all models have been trained yet.
    """
    import joblib

    # Individual artefacts saved by main.py; fall back to load_forecast_model.pkl
    # for whichever model won the best-model race (usually Random Forest).
    candidates = {
        "Random Forest": [
            os.path.join(MODELS_DIR, "random_forest_model.pkl"),
            os.path.join(MODELS_DIR, "load_forecast_model.pkl"),  # fallback
        ],
        "XGBoost": [os.path.join(MODELS_DIR, "xgboost_model.pkl")],
        "SVR":     [os.path.join(MODELS_DIR, "svr_model.pkl")],
    }
    loaded = {}
    for name, paths in candidates.items():
        for path in paths:
            if os.path.exists(path):
                loaded[name] = joblib.load(path)
                print(f"  [comparison] Loaded {name} <- {path}")
                break
        else:
            print(f"  [comparison] {name} not found — skipping.")
    return loaded


def _load_lstm_model():
    """
    Load the LSTM from models/lstm_model.pt.
    Returns (model, y_scaler) or (None, None) if unavailable.
    """
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.pt")
    if not os.path.exists(lstm_path):
        print("  [comparison] lstm_model.pt not found — skipping LSTM.")
        return None, None
    try:
        import sys
        sys.path.insert(0, _ROOT)
        from src.lstm_model import load_lstm
        model, y_scaler = load_lstm(lstm_path)
        print(f"  [comparison] Loaded LSTM <- {lstm_path}")
        return model, y_scaler
    except Exception as exc:
        print(f"  [comparison] LSTM load failed ({exc}) — skipping.")
        return None, None


def _collect_metrics(X_test, y_test, target_scaler=None) -> dict:
    """
    Run inference with every available saved model and return
    {model_name: {metric: value}} without re-training anything.

    If target_scaler is provided, predictions and y_test are inverse-transformed
    to original MW units before computing metrics — ensuring consistency with
    the metrics reported during training.
    """
    import sys
    sys.path.insert(0, _ROOT)

    y_test_arr = np.asarray(y_test)

    # Inverse-transform ground truth to MW once
    if target_scaler is not None:
        y_test_mw = inverse_transform_predictions(y_test_arr, target_scaler)
    else:
        y_test_mw = y_test_arr

    results = {}

    # ── sklearn pipelines (RF / XGBoost / SVR) ───────────────────────────────
    for name, pipeline in _load_sklearn_models().items():
        y_pred_scaled = pipeline.predict(X_test)
        y_pred_mw = (
            inverse_transform_predictions(y_pred_scaled, target_scaler)
            if target_scaler is not None else y_pred_scaled
        )
        results[name] = _compute_four_metrics(y_test_mw, y_pred_mw)
        m = results[name]
        print(f"  {name:<16}  RMSE={m['RMSE (MW)']:.4f}  MAE={m['MAE (MW)']:.4f}"
              f"  R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.2f}%")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    model, y_scaler = _load_lstm_model()
    if model is not None:
        from src.lstm_model import predict_lstm, WINDOW_SIZE
        # X_test and y_test are both scaled from preprocessing
        # predict_lstm returns MW-scale predictions (uses its own y_scaler)
        y_pred_lstm_mw = predict_lstm(model, y_scaler, np.asarray(X_test), device="cpu")
        # y_test is scaled [0,1] from preprocessing — inverse-transform to MW
        y_test_arr = np.asarray(y_test)
        y_test_mw_full = inverse_transform_predictions(y_test_arr, target_scaler) if target_scaler else y_test_arr
        # Align to LSTM's window offset
        y_true_lstm_mw = y_test_mw_full[WINDOW_SIZE:]
        results["LSTM"] = _compute_four_metrics(y_true_lstm_mw, y_pred_lstm_mw)
        m = results["LSTM"]
        print(f"  {'LSTM':<16}  RMSE={m['RMSE (MW)']:.4f}  MAE={m['MAE (MW)']:.4f}"
              f"  R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.2f}%")

    return results


def _build_latex_table(results: dict, caption: str, label: str) -> str:
    """
    Render a publication-ready booktabs LaTeX table.
    Best value per metric is wrapped in \\textbf{}.
    R² uses 4 decimal places; all other metrics use 2.
    """
    df      = pd.DataFrame(results).T
    best    = _best_per_metric(df)
    metrics = list(df.columns)

    def _fmt_cell(value: float, metric: str) -> str:
        dp        = 4 if metric == "R²" else 2
        formatted = f"{value:.{dp}f}"
        if abs(value - best[metric]) < 1e-9:
            return r"\textbf{" + formatted + r"}"
        return formatted

    col_spec = "l" + "r" * len(metrics)
    header   = " & ".join(
        [r"\textbf{Model}"] +
        [r"\textbf{" + _LATEX_HEADERS[m] + r"}" for m in metrics]
    )
    rows = [
        " & ".join([name] + [_fmt_cell(row[m], m) for m in metrics]) + r" \\"
        for name, row in df.iterrows()
    ]

    return "\n".join([
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{" + caption + r"}",
        r"  \label{" + label + r"}",
        r"  \begin{tabular}{" + col_spec + r"}",
        r"    \toprule",
        "    " + header + r" \\",
        r"    \midrule",
        *["    " + r for r in rows],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}",
        r"    \small",
        r"    \item Bold = best result per metric.",
        r"    \item $\downarrow$ lower is better; $\uparrow$ higher is better.",
        r"    \item Dataset: DUQ hourly load (2005--2018), 80/20 chronological split.",
        r"  \end{tablenotes}",
        r"\end{table}",
    ])


def _plot_comparison_chart(results: dict, save_path: str) -> None:
    """
    Grouped bar chart — one subplot per metric, one bar per model.
    Best bar per panel gets a gold border + ★ annotation.
    Raw values are printed on each bar.
    """
    df      = pd.DataFrame(results).T
    metrics = list(df.columns)
    models  = list(df.index)
    best    = _best_per_metric(df)
    colors  = [_MODEL_COLORS.get(m, "#888888") for m in models]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5.2))
    if len(metrics) == 1:
        axes = [axes]

    x     = np.arange(len(models))
    width = 0.58

    for ax, metric in zip(axes, metrics):
        values   = df[metric].values.astype(float)
        val_span = values.max() - values.min() if values.max() != values.min() else 1.0
        bars     = ax.bar(x, values, width=width, color=colors,
                          alpha=0.88, edgecolor="white", linewidth=0.6)

        best_idx = int(np.argmax(values) if _HIGHER_BETTER[metric] else np.argmin(values))

        for i, (bar, val) in enumerate(zip(bars, values)):
            is_best  = (i == best_idx)
            dp       = 4 if metric == "R²" else 2
            label_y  = bar.get_height() + val_span * 0.025
            ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                    f"{val:.{dp}f}",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if is_best else "normal")
            if is_best:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + val_span * 0.10,
                        "★", ha="center", va="bottom",
                        fontsize=11, color="goldenrod")

        # Gold border on best bar
        bars[best_idx].set_edgecolor("goldenrod")
        bars[best_idx].set_linewidth(2.2)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=22, ha="right", fontsize=9)
        ax.set_title(_CHART_SUBTITLES[metric], fontsize=9, pad=6)
        ax.set_ylabel(metric, fontsize=8)
        ax.grid(axis="y", alpha=0.28, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

        # Y-axis padding so annotations don't clip
        ax.set_ylim(0, values.max() + val_span * 0.28)

    # Shared legend
    handles = [Patch(facecolor=_MODEL_COLORS.get(m, "#888888"), label=m, alpha=0.88)
               for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models),
               fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle(
        "Model Performance Comparison — DUQ Hourly Load Forecasting\n"
        "(★ = best per metric)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [comparison] Chart saved -> {save_path}")
    if os.name == "nt":
        os.startfile(save_path)


def _print_console_table(results: dict) -> None:
    """Pretty-print the comparison table to stdout with ★ on best values."""
    df   = pd.DataFrame(results).T
    best = _best_per_metric(df)
    sep  = "=" * 72

    print(f"\n{sep}")
    print(f"  {'Model':<16} {'RMSE (MW)':>11} {'MAE (MW)':>10} {'R²':>8} {'MAPE (%)':>10}")
    print("-" * 72)
    for model, row in df.iterrows():
        def _mark(col):
            v    = row[col]
            star = " ★" if abs(v - best[col]) < 1e-9 else "  "
            dp   = 4 if col == "R²" else 2
            return f"{v:{10}.{dp}f}{star}"
        print(f"  {model:<16}"
              f"{_mark('RMSE (MW)')}"
              f"{_mark('MAE (MW)')}"
              f"{_mark('R²')}"
              f"{_mark('MAPE (%)')}")
    print(sep)
    print("  ★ = best per metric\n")


def build_model_comparison_table(
    X_test,
    y_test,
    target_scaler=None,
    caption: str = (
        "Performance comparison of forecasting models on the DUQ hourly "
        "load dataset (test set, 2005--2018, 80/20 chronological split). "
        "Best result per metric shown in bold."
    ),
    label: str = "tab:model_comparison",
) -> pd.DataFrame:
    """
    Load all saved model artefacts, run inference on (X_test, y_test),
    compute RMSE / MAE / R² / MAPE, then produce:

      results/paper_table.tex        — booktabs LaTeX table, best bold
      results/paper_table.csv        — raw numbers
      results/paper_comparison.png   — grouped bar chart (4 metrics × N models)

    Parameters
    ----------
    X_test  : array-like or pd.DataFrame — scaled test features
    y_test  : array-like                 — raw test targets (MW)
    caption : LaTeX caption string
    label   : LaTeX label string

    Returns
    -------
    pd.DataFrame  rows = models, cols = metrics
    """
    print("\n" + "=" * 60)
    print("  PUBLICATION MODEL COMPARISON TABLE")
    print("=" * 60)

    results = _collect_metrics(X_test, y_test, target_scaler=target_scaler)

    if not results:
        print("  [comparison] No model artefacts found — run main.py first.")
        return pd.DataFrame()

    _print_console_table(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex    = _build_latex_table(results, caption=caption, label=label)
    tex_path = os.path.join(RESULTS_DIR, "paper_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  [comparison] LaTeX table saved -> {tex_path}")
    print("\n" + "─" * 60)
    print(latex)
    print("─" * 60)

    # ── CSV ───────────────────────────────────────────────────────────────────
    df       = pd.DataFrame(results).T
    csv_path = os.path.join(RESULTS_DIR, "paper_table.csv")
    df.to_csv(csv_path)
    print(f"  [comparison] CSV saved -> {csv_path}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    chart_path = os.path.join(RESULTS_DIR, "paper_comparison.png")
    _plot_comparison_chart(results, chart_path)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Quick test
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, _ROOT)
    from src.preprocessing import preprocess

    print("Loading data...")
    X_train, X_test, y_train, y_test, scaler, target_scaler, train_df, test_df = \
        preprocess(os.path.join(_ROOT, "dataset", "DUQ_hourly.csv"))

    # Layer 1 — single model performance demo
    np.random.seed(0)
    y_demo = np.asarray(y_test) + np.random.normal(0, 18, len(y_test))
    ml_metrics = evaluate_model_performance(y_test, y_demo, save_plot=True)
    print("\nLayer 1 metrics:", ml_metrics)

    # Layer 2 — GOA KPI demo
    price = test_df["tou_price"].values[:len(y_demo)]
    after = y_demo * np.random.uniform(0.85, 0.98, len(y_demo))
    compare_before_after(y_demo, after, price)

    # Layer 3 — publication comparison (requires trained model files)
    df_results = build_model_comparison_table(X_test, y_test, target_scaler=target_scaler)
    if not df_results.empty:
        print("\nComparison DataFrame:\n", df_results)
