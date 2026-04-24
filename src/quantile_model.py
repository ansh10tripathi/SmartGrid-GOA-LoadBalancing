"""
src/quantile_model.py
---------------------
Quantile regression for prediction intervals on DUQ hourly load.

Three GradientBoostingRegressor models are trained with the pinball
(quantile) loss at q = 0.10, 0.50, 0.90, producing:
  - lower  : 10th-percentile bound
  - median : 50th-percentile point forecast
  - upper  : 90th-percentile bound

The 80% prediction interval is [lower, upper].

Design notes
------------
- Uses the same X_train / X_test produced by preprocessing.preprocess()
  (MinMaxScaler-normalised, 80/20 chronological split) — no extra scaling.
- Hyperparameters are fixed (no CV search) to keep runtime short; the
  GBR quantile loss is already well-behaved at these settings.
- Predictions and coverage stats are saved to results/quantile_preds.npz
  so the Streamlit dashboard can load them without re-running training.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics  import mean_squared_error, mean_absolute_error, r2_score
import joblib

_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_SRC_DIR)
RESULTS_DIR = os.path.join(_ROOT, "results")
MODELS_DIR  = os.path.join(_ROOT, "models")
PREDS_PATH  = os.path.join(RESULTS_DIR, "quantile_preds.npz")

QUANTILES   = [0.10, 0.50, 0.90]
_GBR_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    min_samples_leaf  = 10,
    random_state      = 42,
)


# ── 1. Train ──────────────────────────────────────────────────────────────────

def train_quantile_models(X_train: np.ndarray,
                          y_train: np.ndarray) -> dict:
    """
    Fit one GradientBoostingRegressor per quantile.
    Returns dict  {0.10: model, 0.50: model, 0.90: model}.
    """
    models = {}
    for q in QUANTILES:
        print(f"[quantile] Training GBR  q={q} ...")
        m = GradientBoostingRegressor(loss="quantile", alpha=q, **_GBR_PARAMS)
        m.fit(X_train, y_train)
        models[q] = m
        print(f"[quantile]   done  q={q}")
    return models


# ── 2. Predict ────────────────────────────────────────────────────────────────

def predict_quantiles(models: dict,
                      X: np.ndarray) -> dict:
    """Returns {0.10: array, 0.50: array, 0.90: array}."""
    return {q: models[q].predict(X) for q in QUANTILES}


# ── 3. Coverage & metrics ─────────────────────────────────────────────────────

def coverage_rate(y_true: np.ndarray,
                  lower:  np.ndarray,
                  upper:  np.ndarray) -> float:
    """Fraction of actuals that fall inside [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of the prediction interval."""
    return float(np.mean(upper - lower))


def evaluate_quantiles(y_true:  np.ndarray,
                       preds:   dict) -> dict:
    """
    Report:
      - RMSE / MAE / R² for the median (q=0.50) forecast
      - 80% interval coverage rate  (target ≥ 0.80)
      - Mean interval width  (narrower = more useful)
    """
    y_med  = preds[0.50]
    rmse   = np.sqrt(mean_squared_error(y_true, y_med))
    mae    = mean_absolute_error(y_true, y_med)
    r2     = r2_score(y_true, y_med)
    cov    = coverage_rate(y_true, preds[0.10], preds[0.90])
    width  = interval_width(preds[0.10], preds[0.90])

    metrics = {
        "RMSE":             round(rmse,  4),
        "MAE":              round(mae,   4),
        "R2":               round(r2,    4),
        "coverage_80pct":   round(cov,   4),
        "mean_interval_MW": round(width, 4),
    }

    print("\n" + "=" * 52)
    print("        QUANTILE REGRESSION RESULTS")
    print("=" * 52)
    print(f"  Median forecast  RMSE : {rmse:>10.4f} MW")
    print(f"  Median forecast  MAE  : {mae:>10.4f} MW")
    print(f"  Median forecast  R²   : {r2:>10.4f}")
    print(f"  80% interval coverage : {cov*100:>9.2f}%"
          f"  {'✓ meets 80% target' if cov >= 0.80 else '✗ below 80% target'}")
    print(f"  Mean interval width   : {width:>10.4f} MW")
    print("=" * 52)
    return metrics


# ── 4. Plot ───────────────────────────────────────────────────────────────────

def plot_ribbon(y_true:   np.ndarray,
                preds:    dict,
                metrics:  dict,
                n_hours:  int = 336,
                save_path: str | None = None) -> None:
    """
    Plot actual load with the 10–90% prediction ribbon for the first
    n_hours of the test set (default 336 h = 2 weeks).
    """
    n   = min(n_hours, len(y_true))
    idx = np.arange(n)

    lo  = preds[0.10][:n]
    med = preds[0.50][:n]
    hi  = preds[0.90][:n]
    act = y_true[:n]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(idx, lo, hi,
                    alpha=0.25, color="steelblue",
                    label="80% prediction interval (10–90%)")
    ax.plot(idx, act, color="black",     linewidth=1.3, label="Actual load")
    ax.plot(idx, med, color="steelblue", linewidth=1.0,
            linestyle="--", label=f"Median forecast  (R²={metrics['R2']:.4f})")
    ax.plot(idx, lo,  color="steelblue", linewidth=0.6, linestyle=":")
    ax.plot(idx, hi,  color="steelblue", linewidth=0.6, linestyle=":")

    ax.set_title(
        f"Quantile Regression — 80% Prediction Interval  "
        f"(coverage={metrics['coverage_80pct']*100:.1f}%,  "
        f"mean width={metrics['mean_interval_MW']:.1f} MW)",
        fontsize=11,
    )
    ax.set_xlabel("Hour (test set)")
    ax.set_ylabel("Load (MW)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = save_path or os.path.join(RESULTS_DIR, "quantile_ribbon.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[quantile] Ribbon plot saved -> {path}")
    if os.name == "nt":
        os.startfile(path)


# ── 5. Save / Load ────────────────────────────────────────────────────────────

def save_quantile_models(models: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for q, m in models.items():
        path = os.path.join(MODELS_DIR, f"quantile_gbr_{int(q*100):02d}.pkl")
        joblib.dump(m, path)
        print(f"[quantile] Model saved -> {path}")


def load_quantile_models() -> dict:
    models = {}
    for q in QUANTILES:
        path = os.path.join(MODELS_DIR, f"quantile_gbr_{int(q*100):02d}.pkl")
        models[q] = joblib.load(path)
    print("[quantile] Models loaded.")
    return models


def save_predictions(y_true: np.ndarray, preds: dict, metrics: dict) -> None:
    """Persist predictions + metrics so the dashboard can load them instantly."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        PREDS_PATH,
        y_true   = y_true,
        lower    = preds[0.10],
        median   = preds[0.50],
        upper    = preds[0.90],
        coverage = np.array([metrics["coverage_80pct"]]),
        width    = np.array([metrics["mean_interval_MW"]]),
        r2       = np.array([metrics["R2"]]),
        rmse     = np.array([metrics["RMSE"]]),
        mae      = np.array([metrics["MAE"]]),
    )
    print(f"[quantile] Predictions saved -> {PREDS_PATH}")


def load_predictions() -> dict | None:
    """Load saved predictions; returns None if file not found."""
    if not os.path.exists(PREDS_PATH):
        return None
    d = np.load(PREDS_PATH)
    return {k: d[k] for k in d.files}


# ── 6. Master pipeline ────────────────────────────────────────────────────────

def run_quantile_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                           X_test:  np.ndarray, y_test:  np.ndarray,
                           target_scaler=None) -> dict:
    """
    Train → predict → evaluate → plot → save.

    If target_scaler is provided, metrics and saved predictions are
    inverse-transformed to original MW units for consistency with the
    paper table and app display.
    Returns metrics dict.
    """
    models  = train_quantile_models(X_train, y_train)
    preds   = predict_quantiles(models, X_test)

    if target_scaler is not None:
        from src.evaluation import inverse_transform_predictions
        y_test_eval  = inverse_transform_predictions(y_test, target_scaler)
        preds_eval   = {q: inverse_transform_predictions(p, target_scaler)
                        for q, p in preds.items()}
    else:
        y_test_eval = y_test
        preds_eval  = preds

    metrics = evaluate_quantiles(y_test_eval, preds_eval)

    plot_ribbon(y_test_eval, preds_eval, metrics)
    save_quantile_models(models)
    save_predictions(y_test_eval, preds_eval, metrics)

    return metrics


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, _ROOT)
    from src.preprocessing import preprocess

    X_train, X_test, y_train, y_test, scaler, train_df, test_df = \
        preprocess(os.path.join(_ROOT, "dataset", "DUQ_hourly.csv"))

    metrics = run_quantile_pipeline(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
    )
    print("\nQuantile metrics:", metrics)
