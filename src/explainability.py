"""
src/explainability.py
---------------------
SHAP-based model explainability for RF, XGBoost, and SVR pipelines.

Explainer strategy
------------------
  RandomForest  : shap.TreeExplainer  — exact, fast (~seconds on full test set)
  XGBoost       : shap.TreeExplainer  — exact, fast
  SVR           : shap.KernelExplainer — model-agnostic, slow O(n²);
                  uses k-means(100) background + 200-sample explain subset

Pipeline handling
-----------------
  All three models are sklearn Pipeline(StandardScaler → estimator).
  TreeExplainer must receive the raw estimator (pipeline["model"]), and the
  data passed to it must be StandardScaler-transformed — i.e. what the tree
  actually saw during training.  The MinMaxScaler-normalised X_train / X_test
  from preprocessing.py is therefore passed through pipeline["scaler"] before
  being handed to the explainer.

Outputs saved to results/
--------------------------
  shap_summary_<model>.png   — mean |SHAP| bar chart, all features ranked
  shap_waterfall_<model>.png — single-prediction waterfall (peak load hour)
  shap_values.npz            — raw SHAP arrays for all three models
                               (loaded by the Streamlit dashboard)
"""

import os
import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SRC_DIR)
RESULTS_DIR = os.path.join(_ROOT, "results")

# Maximum test-set rows explained per model.
_TREE_EXPLAIN_ROWS = 50

# SVR KernelExplainer settings
_SVR_BG_CLUSTERS  = 30
_SVR_EXPLAIN_ROWS = 50


# ── helpers ───────────────────────────────────────────────────────────────────

def _scale(pipeline, X: np.ndarray) -> np.ndarray:
    """Apply the pipeline's StandardScaler step to X (numpy array)."""
    return pipeline["scaler"].transform(X)


def _peak_idx(y_test: np.ndarray) -> int:
    """Index of the highest actual load in the test set."""
    return int(np.argmax(y_test))


# ── 1. Compute SHAP values ────────────────────────────────────────────────────

def _shap_tree(pipeline, X_test: np.ndarray):
    """
    Fast TreeExplainer with NO background data.
    tree_path_dependent needs no background matrix — it computes
    expected_value analytically from the tree structure itself.
    Explains _TREE_EXPLAIN_ROWS evenly-spaced test rows.
    """
    X_te_sc = _scale(pipeline, X_test)

    n     = len(X_te_sc)
    idx   = np.linspace(0, n - 1, min(_TREE_EXPLAIN_ROWS, n), dtype=int)
    X_sub = X_te_sc[idx]

    print(f"    explaining {len(X_sub)} rows...", flush=True)
    # No data= argument — tree_path_dependent is the default and needs none.
    # This is the only path that doesn't allocate a background kernel matrix.
    explainer = shap.TreeExplainer(pipeline["model"])
    sv = explainer.shap_values(X_sub, check_additivity=False)
    if isinstance(sv, list):   # RF regression returns list with one array
        sv = sv[0]
    ev = float(np.atleast_1d(explainer.expected_value)[0])
    print(f"    done.", flush=True)
    return sv, ev, X_sub


def _shap_svr(pipeline, X_train: np.ndarray, X_test: np.ndarray):
    """
    PermutationExplainer for SVR — linear time, no kernel matrix.
    Uses 30 background rows and explains _SVR_EXPLAIN_ROWS test rows.
    """
    X_tr_sc = _scale(pipeline, X_train)
    X_te_sc = _scale(pipeline, X_test)

    bg_idx = np.linspace(0, len(X_tr_sc) - 1, _SVR_BG_CLUSTERS, dtype=int)
    bg     = X_tr_sc[bg_idx]

    n     = len(X_te_sc)
    idx   = np.linspace(0, n - 1, min(_SVR_EXPLAIN_ROWS, n), dtype=int)
    X_sub = X_te_sc[idx]

    print(f"  [SHAP/SVR] PermutationExplainer on {len(X_sub)} rows...", flush=True)
    explainer = shap.PermutationExplainer(pipeline["model"].predict, bg)
    sv_obj    = explainer(X_sub, max_evals=2 * X_sub.shape[1] + 1)
    sv = sv_obj.values
    ev = float(np.atleast_1d(sv_obj.base_values)[0])
    print(f"  [SHAP/SVR] done.", flush=True)
    return sv, ev, X_sub


def compute_shap_values(
    rf_pipeline,
    svr_pipeline,
    xgb_pipeline,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_names: list,
) -> dict:
    """
    Compute SHAP values for all three models.

    Returns a dict keyed by model name, each value is:
      {
        "shap_values":    np.ndarray  (n_samples, n_features),
        "expected_value": float,
        "X_scaled":       np.ndarray  (n_samples, n_features),  # StandardScaled
        "peak_idx":       int,
      }
    """
    peak = _peak_idx(y_test)
    # peak_idx for tree models is clamped to the subsample size
    tree_peak = min(peak, _TREE_EXPLAIN_ROWS - 1)

    results = {}

    print("\n[explainability] Computing SHAP — RandomForest (TreeExplainer)...")
    try:
        sv_rf, ev_rf, Xs_rf = _shap_tree(rf_pipeline, X_test)
        results["RandomForest"] = {
            "shap_values":    sv_rf,
            "expected_value": ev_rf,
            "X_scaled":       Xs_rf,
            "peak_idx":       tree_peak,
        }
        print(f"  [SHAP/RF]  done — shape {sv_rf.shape}")
    except BaseException:
        print("  [SHAP/RF]  FAILED:"); traceback.print_exc()

    print("\n[explainability] Computing SHAP — XGBoost (TreeExplainer)...")
    try:
        sv_xgb, ev_xgb, Xs_xgb = _shap_tree(xgb_pipeline, X_test)
        results["XGBoost"] = {
            "shap_values":    sv_xgb,
            "expected_value": ev_xgb,
            "X_scaled":       Xs_xgb,
            "peak_idx":       tree_peak,
        }
        print(f"  [SHAP/XGB] done — shape {sv_xgb.shape}")
    except BaseException:
        print("  [SHAP/XGB] FAILED:"); traceback.print_exc()

    print("\n[explainability] Computing SHAP — SVR (KernelExplainer)...")
    try:
        sv_svr, ev_svr, Xs_svr = _shap_svr(svr_pipeline, X_train, X_test)
        svr_peak = min(peak, len(sv_svr) - 1)
        results["SVR"] = {
            "shap_values":    sv_svr,
            "expected_value": float(np.atleast_1d(ev_svr)[0]),
            "X_scaled":       Xs_svr,
            "peak_idx":       svr_peak,
        }
        print(f"  [SHAP/SVR] done — shape {sv_svr.shape}")
    except BaseException:
        print("  [SHAP/SVR] FAILED:"); traceback.print_exc()

    return results


# ── 2. Summary bar chart ──────────────────────────────────────────────────────

def plot_summary_bar(shap_values: np.ndarray,
                     feature_names: list,
                     model_name: str,
                     save: bool = True) -> plt.Figure:
    """
    Horizontal bar chart of mean |SHAP| per feature, sorted descending.
    Saved to results/shap_summary_<model>.png.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)          # ascending → bottom = least important
    sorted_names  = [feature_names[i] for i in order]
    sorted_values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_names, sorted_values,
                   color="steelblue", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value| (MW)")
    ax.set_title(f"SHAP Feature Importance — {model_name}\n"
                 f"(mean absolute SHAP over {len(shap_values):,} test samples)")
    ax.grid(axis="x", alpha=0.3)

    # Annotate top-5 bars
    for bar, val in zip(bars[-5:], sorted_values[-5:]):
        ax.text(val + mean_abs.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"shap_summary_{model_name.lower()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [SHAP] Summary bar saved -> {path}")
        if os.name == "nt":
            os.startfile(path)

    return fig


# ── 3. Waterfall plot ─────────────────────────────────────────────────────────

def plot_waterfall(shap_values:    np.ndarray,
                   expected_value: float,
                   X_scaled:       np.ndarray,
                   feature_names:  list,
                   sample_idx:     int,
                   model_name:     str,
                   y_true:         float | None = None,
                   save:           bool = True) -> plt.Figure:
    """
    Waterfall chart for a single prediction.

    Shows how each feature pushes the prediction above or below the
    model's expected (baseline) output.  Positive SHAP = pushes load up,
    negative = pushes load down.

    Uses shap.Explanation + shap.plots.waterfall when available (shap>=0.42),
    falls back to a pure-matplotlib implementation otherwise.
    """
    sv_row = shap_values[sample_idx]        # (n_features,)
    x_row  = X_scaled[sample_idx]           # (n_features,) — StandardScaled values
    pred   = expected_value + sv_row.sum()

    title_suffix = (f"  |  actual={y_true:.1f} MW" if y_true is not None else "")

    # ── Try native shap waterfall ─────────────────────────────────────────────
    # shap.plots.waterfall always creates its own figure internally;
    # we call it with show=False then grab plt.gcf() to get that figure.
    try:
        exp = shap.Explanation(
            values        = sv_row,
            base_values   = expected_value,
            data          = x_row,
            feature_names = feature_names,
        )
        shap.plots.waterfall(exp, max_display=15, show=False)
        fig = plt.gcf()
        fig.set_size_inches(9, 6)
        fig.suptitle(
            f"SHAP Waterfall — {model_name}  |  pred={pred:.1f} MW{title_suffix}",
            fontsize=10, y=1.01,
        )
        plt.tight_layout()

    except Exception:
        # ── Matplotlib fallback ───────────────────────────────────────────────
        fig = _waterfall_mpl(sv_row, expected_value, feature_names,
                             model_name, pred, title_suffix)

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"shap_waterfall_{model_name.lower()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [SHAP] Waterfall saved -> {path}")
        if os.name == "nt":
            os.startfile(path)

    return fig


def _waterfall_mpl(sv_row:         np.ndarray,
                   expected_value: float,
                   feature_names:  list,
                   model_name:     str,
                   pred:           float,
                   title_suffix:   str) -> plt.Figure:
    """Pure-matplotlib waterfall — top-15 features by |SHAP|."""
    top_n   = 15
    order   = np.argsort(np.abs(sv_row))[-top_n:][::-1]   # descending |SHAP|
    names   = [feature_names[i] for i in order]
    values  = sv_row[order]

    # Build running totals for the waterfall bars
    running = expected_value
    lefts, heights, colors = [], [], []
    for v in values:
        lefts.append(running if v >= 0 else running + v)
        heights.append(abs(v))
        colors.append("tomato" if v >= 0 else "steelblue")
        running += v

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(top_n), heights, left=lefts, color=colors, alpha=0.85)
    ax.axvline(expected_value, color="black", linewidth=1.0,
               linestyle="--", label=f"E[f(x)] = {expected_value:.1f}")
    ax.axvline(pred, color="purple", linewidth=1.2,
               linestyle="-", label=f"f(x) = {pred:.1f}")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("SHAP value (MW)")
    ax.set_title(
        f"SHAP Waterfall — {model_name}  |  pred={pred:.1f} MW{title_suffix}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ── 4. Save / Load ────────────────────────────────────────────────────────────

_NPZ_PATH = os.path.join(RESULTS_DIR, "shap_values.npz")


def save_shap(shap_results: dict, feature_names: list) -> None:
    """
    Persist SHAP arrays to results/shap_values.npz.
    Only saves models that are present in shap_results (failed ones are skipped).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {"feature_names": np.array(feature_names)}
    key_map = {
        "RandomForest": ("rf_sv",  "rf_ev"),
        "XGBoost":      ("xgb_sv", "xgb_ev"),
        "SVR":          ("svr_sv", "svr_ev"),
    }
    peak_saved = False
    for model_name, (sv_key, ev_key) in key_map.items():
        if model_name not in shap_results:
            continue
        r = shap_results[model_name]
        payload[sv_key] = r["shap_values"]
        payload[ev_key] = np.array([r["expected_value"]])
        if not peak_saved:
            payload["peak_idx"] = np.array([r["peak_idx"]])
            peak_saved = True
    np.savez(_NPZ_PATH, **payload)
    print(f"  [SHAP] Values saved -> {_NPZ_PATH}")


def load_shap() -> dict | None:
    """Load saved SHAP data; returns None if file not found."""
    if not os.path.exists(_NPZ_PATH):
        return None
    d = np.load(_NPZ_PATH, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ── 5. Master pipeline ────────────────────────────────────────────────────────

def run_explainability_pipeline(
    rf_pipeline,
    svr_pipeline,
    xgb_pipeline,
    X_train:       np.ndarray,
    X_test:        np.ndarray,
    y_test:        np.ndarray,
    feature_names: list,
) -> None:
    """
    Compute SHAP for all three models, generate and save all plots.
    Called from main.py after compare_models().
    """
    shap_results = compute_shap_values(
        rf_pipeline, svr_pipeline, xgb_pipeline,
        X_train, X_test, y_test, feature_names,
    )

    peak = shap_results["RandomForest"]["peak_idx"]

    for name in ("RandomForest", "XGBoost", "SVR"):
        if name not in shap_results:
            print(f"  [explainability] Skipping plots for {name} (SHAP failed).")
            continue
        r = shap_results[name]
        plot_summary_bar(r["shap_values"], feature_names, name)
        plot_waterfall(
            r["shap_values"], r["expected_value"], r["X_scaled"],
            feature_names,
            sample_idx = r["peak_idx"],
            model_name = name,
            y_true     = float(y_test[r["peak_idx"]]),
        )

    save_shap(shap_results, feature_names)
    print("\n[explainability] All SHAP plots and values saved.")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, _ROOT)
    from src.preprocessing     import preprocess, FEATURE_COLS
    from src.forecasting_model import (train_random_forest, train_xgboost)
    from sklearn.svm import SVR as _SVR
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.pipeline import Pipeline as _Pipeline

    X_train, X_test, y_train, y_test, scaler, train_df, test_df = \
        preprocess(os.path.join(_ROOT, "dataset", "DUQ_hourly.csv"))

    print("\nTraining models (use_search=False for quick test)...")
    rf  = train_random_forest(X_train, y_train, use_search=False)
    xgb = train_xgboost(X_train,       y_train, use_search=False)

    # SVR capped at 8k rows — O(n²) kernel would hang on full 95k
    _n   = len(X_train)
    _cap = 8_000
    svr  = _Pipeline([("scaler", _SS()),
                      ("model",  _SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))])
    svr.fit(X_train.iloc[max(0, _n - _cap):], y_train.iloc[max(0, _n - _cap):])
    print(f"[explainability] SVR fitted on last {min(_cap, _n):,} rows.")

    run_explainability_pipeline(
        rf, svr, xgb,
        X_train.values, X_test.values, y_test.values,
        FEATURE_COLS,
    )
