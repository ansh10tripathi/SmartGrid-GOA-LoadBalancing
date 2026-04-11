"""
src/paper_comparison.py
-----------------------
Publication-ready model comparison table and chart for the DUQ dataset.

Metrics reported
----------------
  RMSE  — Root Mean Square Error (MW)          lower is better
  MAE   — Mean Absolute Error (MW)             lower is better
  R²    — Coefficient of Determination         higher is better
  MAPE  — Mean Absolute Percentage Error (%)   lower is better

Model loading strategy
----------------------
  Random Forest  }
  SVR            }  re-trained with fixed (non-search) params for speed;
  XGBoost        }  results are deterministic and reproducible.

  LSTM           — loaded from models/lstm_model.pt if it exists;
                   skipped with a warning if not found (torch optional).

  The same preprocessing pipeline (80/20 chronological split,
  MinMaxScaler fit on train only) is used for all models so metrics
  are directly comparable.

Outputs
-------
  results/paper_comparison.png   — grouped bar chart (4 metrics × N models)
  results/paper_table.tex        — LaTeX booktabs table, best per metric bold
  results/paper_table.csv        — raw numbers for reference
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SRC_DIR)
RESULTS_DIR = os.path.join(_ROOT, "results")
MODELS_DIR  = os.path.join(_ROOT, "models")

sys.path.insert(0, _ROOT)


# ── 1. Metrics ────────────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Rows where |y_true| < 1e-6 are excluded to avoid division by zero
    (load is always >> 0 for DUQ, so this only guards against edge cases).
    """
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = _mape(y_true, y_pred)
    return {
        "RMSE (MW)": round(rmse, 4),
        "MAE (MW)":  round(mae,  4),
        "R²":        round(r2,   4),
        "MAPE (%)":  round(mape, 4),
    }


# ── 2. Load data & models ─────────────────────────────────────────────────────

def _load_data():
    from src.preprocessing import preprocess
    return preprocess(os.path.join(_ROOT, "dataset", "DUQ_hourly.csv"))


def _infer_sklearn(pipeline, X_test) -> np.ndarray:
    return pipeline.predict(X_test)


def _infer_lstm(X_test: np.ndarray, y_test: np.ndarray):
    """
    Load lstm_model.pt and run inference.
    Returns (y_pred_aligned, y_true_aligned) trimmed to [WINDOW_SIZE:].
    Returns (None, None) if torch or the model file is unavailable.
    """
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.pt")
    if not os.path.exists(lstm_path):
        print("  [paper] lstm_model.pt not found — skipping LSTM.")
        return None, None
    try:
        from src.lstm_model import load_lstm, predict_lstm, WINDOW_SIZE
        model, y_scaler = load_lstm(lstm_path)
        y_pred = predict_lstm(model, y_scaler, X_test, device="cpu")
        return y_pred, y_test[WINDOW_SIZE:]
    except Exception as exc:
        print(f"  [paper] LSTM load/inference failed ({exc}) — skipping.")
        return None, None


# SVR is O(n²/n³) — cap training at this many chronological tail rows.
# 8 k rows gives a representative recent-history sample and fits in ~20 s.
_SVR_TRAIN_CAP = 8_000


def collect_results(X_train, X_test, y_train, y_test) -> dict:
    """
    Train RF / SVR / XGBoost with fixed params (fast, deterministic),
    load LSTM if available, return dict of {model_name: metrics_dict}.

    SVR is capped at _SVR_TRAIN_CAP chronological tail rows to avoid the
    O(n²) memory / O(n³) time blow-up on the full 95 k training set.
    """
    from src.forecasting_model import (train_random_forest, train_svr,
                                       train_xgboost)
    from sklearn.svm import SVR as _SVR
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.pipeline import Pipeline as _Pipeline

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("  [paper] Training Random Forest (use_search=False)...")
    rf_pipe  = train_random_forest(X_train, y_train, use_search=False)
    y_pred   = _infer_sklearn(rf_pipe, X_test)
    results["Random Forest"] = compute_all_metrics(
        np.asarray(y_test), np.asarray(y_pred)
    )
    m = results["Random Forest"]
    print(f"         RMSE={m['RMSE (MW)']:.4f}  MAE={m['MAE (MW)']:.4f}  "
          f"R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.4f}%")

    # ── SVR — capped subsample ────────────────────────────────────────────────
    n_tr  = len(X_train)
    start = max(0, n_tr - _SVR_TRAIN_CAP)
    X_svr = X_train.iloc[start:]
    y_svr = y_train.iloc[start:]
    print(f"  [paper] Training SVR on last {len(X_svr):,} rows "
          f"(capped from {n_tr:,}, use_search=False)...")
    svr_pipe = _Pipeline([
        ("scaler", _SS()),
        ("model",  _SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
    ])
    svr_pipe.fit(X_svr, y_svr)
    y_pred = _infer_sklearn(svr_pipe, X_test)
    results["SVR"] = compute_all_metrics(
        np.asarray(y_test), np.asarray(y_pred)
    )
    m = results["SVR"]
    print(f"         RMSE={m['RMSE (MW)']:.4f}  MAE={m['MAE (MW)']:.4f}  "
          f"R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.4f}%")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("  [paper] Training XGBoost (use_search=False)...")
    xgb_pipe = train_xgboost(X_train, y_train, use_search=False)
    y_pred   = _infer_sklearn(xgb_pipe, X_test)
    results["XGBoost"] = compute_all_metrics(
        np.asarray(y_test), np.asarray(y_pred)
    )
    m = results["XGBoost"]
    print(f"         RMSE={m['RMSE (MW)']:.4f}  MAE={m['MAE (MW)']:.4f}  "
          f"R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.4f}%")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("  [paper] Loading LSTM...")
    y_pred_lstm, y_true_lstm = _infer_lstm(X_test.values, y_test.values)
    if y_pred_lstm is not None:
        results["LSTM"] = compute_all_metrics(y_true_lstm, y_pred_lstm)
        m = results["LSTM"]
        print(f"         RMSE={m['RMSE (MW)']:.4f}  MAE={m['MAE (MW)']:.4f}  "
              f"R²={m['R²']:.4f}  MAPE={m['MAPE (%)']:.4f}%")

    return results


# ── 3. LaTeX table ────────────────────────────────────────────────────────────

# Direction: True = higher is better, False = lower is better
_METRIC_HIGHER_BETTER = {
    "RMSE (MW)": False,
    "MAE (MW)":  False,
    "R²":        True,
    "MAPE (%)":  False,
}

_LATEX_METRIC_LABELS = {
    "RMSE (MW)": r"RMSE (MW)$\downarrow$",
    "MAE (MW)":  r"MAE (MW)$\downarrow$",
    "R²":        r"$R^2$$\uparrow$",
    "MAPE (%)":  r"MAPE (\%)$\downarrow$",
}


def _best_per_metric(df: pd.DataFrame) -> dict:
    """
    Returns {metric: best_value} where best is min or max depending on direction.
    """
    best = {}
    for col in df.columns:
        higher = _METRIC_HIGHER_BETTER.get(col, False)
        best[col] = df[col].max() if higher else df[col].min()
    return best


def _fmt(value: float, metric: str, best_value: float) -> str:
    """
    Format a cell value.  Apply \\textbf{} if this value equals the best.
    R² uses 4 decimal places; all others use 2.
    """
    decimals = 4 if metric == "R²" else 2
    formatted = f"{value:.{decimals}f}"
    if abs(value - best_value) < 1e-9:
        return r"\textbf{" + formatted + r"}"
    return formatted


def build_latex_table(results: dict, caption: str, label: str) -> str:
    """
    Build a publication-ready booktabs LaTeX table.

    Parameters
    ----------
    results : {model_name: {metric: value}}
    caption : LaTeX \\caption{} text
    label   : LaTeX \\label{} key

    Returns
    -------
    Complete LaTeX table as a string.
    """
    df   = pd.DataFrame(results).T          # rows = models, cols = metrics
    best = _best_per_metric(df)
    metrics = list(df.columns)

    col_spec = "l" + "r" * len(metrics)
    header   = " & ".join(
        ["\\textbf{Model}"] +
        [r"\textbf{" + _LATEX_METRIC_LABELS[m] + r"}" for m in metrics]
    )

    rows = []
    for model_name, row in df.iterrows():
        cells = [model_name] + [
            _fmt(row[m], m, best[m]) for m in metrics
        ]
        rows.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{" + caption + r"}",
        r"  \label{" + label + r"}",
        r"  \begin{tabular}{" + col_spec + r"}",
        r"    \toprule",
        "    " + header + r" \\",
        r"    \midrule",
    ] + ["    " + r for r in rows] + [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}",
        r"    \small",
        r"    \item Bold values indicate the best result per metric.",
        r"    \item $\downarrow$ lower is better; $\uparrow$ higher is better.",
        r"    \item Dataset: DUQ hourly load (2005--2018), 80/20 chronological split.",
        r"  \end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── 4. Grouped bar chart ──────────────────────────────────────────────────────

# Normalise each metric to [0, 1] for a single readable chart.
# For lower-is-better metrics we invert so that taller bar = better.
_CHART_LABELS = {
    "RMSE (MW)": "RMSE (MW)\n(lower=better)",
    "MAE (MW)":  "MAE (MW)\n(lower=better)",
    "R²":        "R²\n(higher=better)",
    "MAPE (%)":  "MAPE (%)\n(lower=better)",
}

_MODEL_COLORS = {
    "Random Forest": "#4C72B0",
    "SVR":           "#DD8452",
    "XGBoost":       "#55A868",
    "LSTM":          "#C44E52",
}


def plot_comparison_chart(results: dict, save: bool = True) -> plt.Figure:
    """
    Grouped bar chart: one group per metric, one bar per model.
    Raw values are shown on the bars; axes are scaled per metric group
    so all four panels are readable side-by-side.
    """
    df      = pd.DataFrame(results).T
    metrics = list(df.columns)
    models  = list(df.index)
    n_m     = len(metrics)
    n_mod   = len(models)

    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 5))
    if n_m == 1:
        axes = [axes]

    x      = np.arange(n_mod)
    width  = 0.6
    best   = _best_per_metric(df)
    colors = [_MODEL_COLORS.get(m, "#888888") for m in models]

    for ax, metric in zip(axes, metrics):
        values  = df[metric].values.astype(float)
        bars    = ax.bar(x, values, width=width, color=colors,
                         alpha=0.88, edgecolor="white", linewidth=0.6)

        # Annotate each bar with its raw value
        decimals = 4 if metric == "R²" else 2
        for bar, val, model in zip(bars, values, models):
            is_best = abs(val - best[metric]) < 1e-9
            weight  = "bold" if is_best else "normal"
            color   = "black"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (values.max() - values.min()) * 0.02,
                f"{val:.{decimals}f}",
                ha="center", va="bottom", fontsize=8,
                fontweight=weight, color=color,
            )
            # Star marker on best bar
            if is_best:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (values.max() - values.min()) * 0.09,
                    "★", ha="center", va="bottom", fontsize=10, color="goldenrod",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(_CHART_LABELS[metric], fontsize=9, pad=6)
        ax.set_ylabel(metric, fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

        # Highlight best bar with a subtle border
        best_idx = int(np.argmin(values) if not _METRIC_HIGHER_BETTER[metric]
                       else np.argmax(values))
        bars[best_idx].set_edgecolor("goldenrod")
        bars[best_idx].set_linewidth(2.0)

    # Shared legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_MODEL_COLORS.get(m, "#888888"),
                     label=m, alpha=0.88) for m in models]
    fig.legend(handles=handles, loc="lower center",
               ncol=n_mod, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle(
        "Model Performance Comparison — DUQ Hourly Load Forecasting\n"
        "(★ = best per metric)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "paper_comparison.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  [paper] Chart saved -> {path}")
        if os.name == "nt":
            os.startfile(path)

    return fig


# ── 5. Console summary table ──────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    df   = pd.DataFrame(results).T
    best = _best_per_metric(df)

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  {'Model':<16} {'RMSE (MW)':>10} {'MAE (MW)':>10} {'R²':>8} {'MAPE (%)':>10}")
    print("-" * 68)
    for model, row in df.iterrows():
        def mark(col):
            v = row[col]
            star = " ★" if abs(v - best[col]) < 1e-9 else "  "
            decimals = 4 if col == "R²" else 2
            return f"{v:{10}.{decimals}f}{star}"
        print(f"  {model:<16}"
              f"{mark('RMSE (MW)')}"
              f"{mark('MAE (MW)')}"
              f"{mark('R²')}"
              f"{mark('MAPE (%)')}")
    print(sep)
    print("  ★ = best per metric\n")


# ── 6. Master pipeline ────────────────────────────────────────────────────────

def run_paper_comparison(
    caption: str = (
        "Performance comparison of forecasting models on the DUQ hourly "
        "load dataset (test set, 2015--2018). "
        "Best result per metric shown in bold."
    ),
    label: str = "tab:model_comparison",
) -> pd.DataFrame:
    """
    Full pipeline: load data → train/infer → metrics → table → chart.

    Returns the results DataFrame (rows = models, cols = metrics).
    Saves:
      results/paper_comparison.png
      results/paper_table.tex
      results/paper_table.csv
    """
    print("\n[paper_comparison] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, train_df, test_df = _load_data()

    print("\n[paper_comparison] Collecting model results...")
    results = collect_results(X_train, X_test, y_train, y_test)

    print_summary(results)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex = build_latex_table(results, caption=caption, label=label)
    tex_path = os.path.join(RESULTS_DIR, "paper_table.tex")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"  [paper] LaTeX table saved -> {tex_path}")
    print("\n" + "─" * 60)
    print(latex)
    print("─" * 60)

    # ── CSV ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results).T
    csv_path = os.path.join(RESULTS_DIR, "paper_table.csv")
    df.to_csv(csv_path)
    print(f"  [paper] CSV saved -> {csv_path}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    plot_comparison_chart(results)

    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_paper_comparison()
