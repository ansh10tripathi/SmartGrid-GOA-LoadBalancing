"""
src/forecasting_model.py
------------------------
ML pipeline for the DUQ_hourly dataset (~119 K rows).

Models
------
  - Random Forest  (existing, unchanged)
  - SVR            (new)
  - XGBoost        (new)

Best model is selected automatically by highest R² and used for GOA.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from src.evaluation import evaluate_model_performance

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(_SRC_DIR), "models", "load_forecast_model.pkl")


# ── 1. Time-based split ───────────────────────────────────────────────────────

def split_data(X, y, test_size: float = 0.2):
    """Strict chronological split — no shuffling to prevent future-data leakage."""
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"[split_data] Train={len(X_train):,}, Test={len(X_test):,} "
          f"(time-based, no shuffle)")
    return X_train, X_test, y_train, y_test


# ── 2. Train individual models ────────────────────────────────────────────────

def train_random_forest(X_train, y_train, use_search: bool = True,
                        random_state: int = 42):
    """Train a Random Forest (existing logic, unchanged)."""
    if use_search:
        base_rf = RandomForestRegressor(random_state=random_state, n_jobs=1)
        param_dist = {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [15, 20, 25],
            "min_samples_split": [2, 4],
            "min_samples_leaf":  [1, 2],
            "max_features":      ["sqrt", "log2"],
        }
        search = RandomizedSearchCV(
            base_rf, param_distributions=param_dist,
            n_iter=10, cv=3, scoring="r2",
            random_state=random_state, n_jobs=1, verbose=1,
        )
        print("[train_random_forest] Running RandomizedSearchCV (10 iter x 3-fold CV)...")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"[train_random_forest] Best params : {search.best_params_}")
        print(f"[train_random_forest] Best CV R2  : {search.best_score_:.4f}")
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=20,
            min_samples_split=2, min_samples_leaf=1,
            max_features="sqrt", random_state=random_state, n_jobs=1,
        )
        model.fit(X_train, y_train)
        print("[train_random_forest] Trained fixed-param RandomForest (200 trees, depth=20).")
    return model


def train_svr(X_train, y_train):
    """Train SVR with StandardScaler (SVR is sensitive to feature scale)."""
    print("[train_svr] Fitting StandardScaler + SVR...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_scaled, y_train)
    print("[train_svr] SVR training complete.")
    # Bundle scaler with model so predict() works transparently
    return {"model": model, "scaler": scaler, "type": "svr"}


def train_xgboost(X_train, y_train, random_state: int = 42):
    """Train XGBoost Regressor."""
    print("[train_xgboost] Training XGBoost...")
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    print("[train_xgboost] XGBoost training complete.")
    return model


# ── 3. Predict helper (handles SVR bundle) ────────────────────────────────────

def _predict(model_obj, X):
    """Return predictions; handles plain sklearn/xgb models and SVR bundles."""
    if isinstance(model_obj, dict) and model_obj.get("type") == "svr":
        X_scaled = model_obj["scaler"].transform(X)
        return model_obj["model"].predict(X_scaled)
    return model_obj.predict(X)


# ── 4. Evaluate a single model ────────────────────────────────────────────────

def evaluate_model(model_obj, X_test, y_test):
    """Return metrics dict + predictions array."""
    y_pred = _predict(model_obj, X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
    print(f"[evaluate_model] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    status = "TARGET MET" if r2 >= 0.95 else "below 0.95"
    print(f"[evaluate_model] R2 >= 0.95 -> {status}")
    return metrics, y_pred


# ── 5. Multi-model comparison ─────────────────────────────────────────────────

def compare_models(X_train, X_test, y_train, y_test, use_search: bool = True):
    """
    Train RF, SVR, XGBoost; evaluate all; print comparison table.
    Returns (best_model_obj, best_metrics, y_test, best_y_pred, X_test,
             all_metrics_dict, all_preds_dict).
    """
    # --- Train ---
    rf_model  = train_random_forest(X_train, y_train, use_search=use_search)
    svr_model = train_svr(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # --- Evaluate ---
    rf_metrics,  y_pred_rf  = evaluate_model(rf_model,  X_test, y_test)
    svr_metrics, y_pred_svr = evaluate_model(svr_model, X_test, y_test)
    xgb_metrics, y_pred_xgb = evaluate_model(xgb_model, X_test, y_test)

    # --- Print comparison table ---
    print("\n" + "=" * 52)
    print(f"  {'Model':<16} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 52)
    print(f"  {'RandomForest':<16} {rf_metrics['RMSE']:>8.4f} {rf_metrics['MAE']:>8.4f} {rf_metrics['R2']:>8.4f}")
    print(f"  {'SVR':<16} {svr_metrics['RMSE']:>8.4f} {svr_metrics['MAE']:>8.4f} {svr_metrics['R2']:>8.4f}")
    print(f"  {'XGBoost':<16} {xgb_metrics['RMSE']:>8.4f} {xgb_metrics['MAE']:>8.4f} {xgb_metrics['R2']:>8.4f}")
    print("=" * 52)

    # --- Select best by R² ---
    candidates = [
        ("RandomForest", rf_model,  rf_metrics,  y_pred_rf),
        ("SVR",          svr_model, svr_metrics, y_pred_svr),
        ("XGBoost",      xgb_model, xgb_metrics, y_pred_xgb),
    ]
    best_name, best_model, best_metrics, best_y_pred = max(
        candidates, key=lambda t: t[2]["R2"]
    )
    print(f"\n[compare_models] Best model: {best_name}  (R²={best_metrics['R2']:.4f})")

    all_metrics = {
        "RandomForest": rf_metrics,
        "SVR":          svr_metrics,
        "XGBoost":      xgb_metrics,
    }
    all_preds = {
        "RandomForest": y_pred_rf,
        "SVR":          y_pred_svr,
        "XGBoost":      y_pred_xgb,
    }
    return best_model, best_metrics, y_test, best_y_pred, X_test, all_metrics, all_preds


# ── 6. Save / Load ────────────────────────────────────────────────────────────

def save_model(model, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Model saved -> {path}")


def load_model(path: str = MODEL_PATH):
    model = joblib.load(path)
    print(f"[load_model] Model loaded <- {path}")
    return model


# ── 7. Master pipeline ────────────────────────────────────────────────────────

def run_forecasting_pipeline(X, y, use_search: bool = True):
    """
    Full pipeline: split -> train all models -> compare -> save best -> plot.
    Returns (best_model, best_metrics, y_test, best_y_pred, X_test,
             all_metrics, all_preds).
    """
    X_train, X_test, y_train, y_test = split_data(X, y)
    best_model, best_metrics, y_test, best_y_pred, X_test, all_metrics, all_preds = \
        compare_models(X_train, X_test, y_train, y_test, use_search=use_search)
    save_model(best_model)
    evaluate_model_performance(y_test, best_y_pred)
    return best_model, best_metrics, y_test, best_y_pred, X_test, all_metrics, all_preds


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import preprocess
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    X, y, scaler, df = preprocess(_os.path.join(_root, "dataset", "DUQ_hourly.csv"))
    best_model, best_metrics, y_test, best_y_pred, X_test, all_metrics, all_preds = \
        run_forecasting_pipeline(X, y, use_search=False)
    print("\nBest model metrics:", best_metrics)
    print("All metrics:", all_metrics)
