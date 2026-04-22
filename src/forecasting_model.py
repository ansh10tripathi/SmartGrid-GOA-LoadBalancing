"""
src/forecasting_model.py
------------------------
ML pipeline for the DUQ_hourly dataset (~119 K rows).

Models
------
  - Random Forest
  - SVR
  - XGBoost

Hyperparameter tuning uses TimeSeriesSplit(n_splits=5) to prevent
future-data leakage that shuffled CV causes on temporal data.

Best model is selected automatically by highest R² and saved for GOA.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from src.evaluation import evaluate_model_performance

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(_SRC_DIR), "models", "load_forecast_model.pkl")

# Shared CV strategy — always train on past, validate on future
_TS_CV = TimeSeriesSplit(n_splits=5)


# ── 1. Train individual models ───────────────────────────────────────────────

def _make_pipeline(estimator) -> Pipeline:
    """Wrap any estimator in StandardScaler → estimator Pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def train_random_forest(X_train, y_train, use_search: bool = True,
                        random_state: int = 42) -> Pipeline:
    """
    Returns a fitted Pipeline(StandardScaler → RandomForest).
    Param keys are prefixed with 'model__' to target the pipeline step.
    """
    pipeline = _make_pipeline(
        RandomForestRegressor(random_state=random_state, n_jobs=1)
    )
    if use_search:
        param_dist = {
            "model__n_estimators":      [100, 200, 300],
            "model__max_depth":         [15, 20, 25],
            "model__min_samples_split": [2, 4],
            "model__min_samples_leaf":  [1, 2],
            "model__max_features":      ["sqrt", "log2"],
        }
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist,
            n_iter=10, cv=_TS_CV, scoring="r2",
            random_state=random_state, n_jobs=1, verbose=1,
        )
        print("[train_random_forest] RandomizedSearchCV with TimeSeriesSplit(n_splits=5)...")
        search.fit(X_train, y_train)
        print(f"[train_random_forest] Best params : {search.best_params_}")
        print(f"[train_random_forest] Best CV R²  : {search.best_score_:.4f}")
        return search.best_estimator_
    pipeline.fit(X_train, y_train)
    print("[train_random_forest] Trained fixed-param RandomForest pipeline.")
    return pipeline


# SVR is O(n²) in memory and O(n³) in time — impractical to CV on 95k rows.
# Search on a chronological subsample, then refit best params on full data.
_SVR_SEARCH_ROWS = 5_000


def train_svr(X_train, y_train, use_search: bool = True) -> Pipeline:
    """
    Returns a fitted Pipeline(StandardScaler → SVR) on the full X_train.

    When use_search=True:
      1. Take the last _SVR_SEARCH_ROWS rows of X_train (chronological tail)
         as the search subset — recent data is most representative.
      2. Run RandomizedSearchCV with TimeSeriesSplit on that subset to find
         best C / gamma / epsilon.
      3. Build a fresh pipeline with those best params and fit it on the
         FULL X_train so the final model sees all training history.

    This keeps CV time to ~2 min instead of hours while still finding
    good hyperparameters.
    """
    if use_search:
        # ── Step 1: subsample for search (chronological tail) ────────────────
        n          = len(X_train)
        sub_X      = X_train.iloc[max(0, n - _SVR_SEARCH_ROWS):]
        sub_y      = y_train.iloc[max(0, n - _SVR_SEARCH_ROWS):]
        print(f"[train_svr] SVR search on {len(sub_X):,} rows "
              f"(last {_SVR_SEARCH_ROWS:,} of {n:,} train rows)...")

        param_dist = {
            "model__C":       [1, 10, 50, 100, 200],
            "model__gamma":   [0.001, 0.01, 0.1, "scale", "auto"],
            "model__epsilon": [0.05, 0.1, 0.2],
        }
        search = RandomizedSearchCV(
            _make_pipeline(SVR(kernel="rbf")),
            param_distributions=param_dist,
            n_iter=10, cv=TimeSeriesSplit(n_splits=3),
            scoring="r2", n_jobs=1, verbose=1,
        )
        search.fit(sub_X, sub_y)
        best_params = {k.replace("model__", ""): v
                       for k, v in search.best_params_.items()}
        print(f"[train_svr] Best params : {search.best_params_}")
        print(f"[train_svr] Best CV R²  : {search.best_score_:.4f}")

        # ── Step 2: refit on full training data with best params ─────────────
        print(f"[train_svr] Refitting on full {n:,} train rows...")
        pipeline = _make_pipeline(SVR(kernel="rbf", **best_params))
        pipeline.fit(X_train, y_train)
        print("[train_svr] Full refit complete.")
        return pipeline

    pipeline = _make_pipeline(SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
    pipeline.fit(X_train, y_train)
    print("[train_svr] Trained fixed-param SVR pipeline.")
    return pipeline


def train_xgboost(X_train, y_train, use_search: bool = True,
                  random_state: int = 42) -> Pipeline:
    """
    Returns a fitted Pipeline(StandardScaler → XGBoost).
    Param keys are prefixed with 'model__' to target the XGBoost step.
    """
    pipeline = _make_pipeline(
        XGBRegressor(random_state=random_state, n_jobs=1, verbosity=0)
    )
    if use_search:
        param_dist = {
            "model__n_estimators":     [200, 300, 400],
            "model__max_depth":        [4, 6, 8],
            "model__learning_rate":    [0.01, 0.05, 0.1],
            "model__subsample":        [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        }
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist,
            n_iter=10, cv=_TS_CV, scoring="r2",
            random_state=random_state, n_jobs=1, verbose=1,
        )
        print("[train_xgboost] RandomizedSearchCV with TimeSeriesSplit(n_splits=5)...")
        search.fit(X_train, y_train)
        print(f"[train_xgboost] Best params : {search.best_params_}")
        print(f"[train_xgboost] Best CV R²  : {search.best_score_:.4f}")
        return search.best_estimator_
    pipeline.fit(X_train, y_train)
    print("[train_xgboost] Trained fixed-param XGBoost pipeline.")
    return pipeline


# ── 2. Evaluate — all models are now plain Pipelines, no special-casing ───────

def evaluate_model(pipeline: Pipeline, X_test, y_test, target_scaler=None):
    """
    Evaluate pipeline on test set.

    If target_scaler is provided, predictions and y_test are inverse-transformed
    to original MW units before computing RMSE/MAE/R2 — matching the scale
    used in the paper table and app display.
    Returns (metrics_dict, y_pred_scaled) where y_pred_scaled is in the same
    normalised space as y_test (for downstream GOA / plot alignment).
    """
    from src.evaluation import inverse_transform_predictions
    y_pred_scaled = pipeline.predict(X_test)
    y_test_arr    = np.asarray(y_test)

    if target_scaler is not None:
        y_true_mw = inverse_transform_predictions(y_test_arr,    target_scaler)
        y_pred_mw = inverse_transform_predictions(y_pred_scaled, target_scaler)
        print(f"[evaluate_model] DEBUG scale=MW  "
              f"y_true=[{y_true_mw.min():.1f}, {y_true_mw.max():.1f}]  "
              f"y_pred=[{y_pred_mw.min():.1f}, {y_pred_mw.max():.1f}]")
    else:
        y_true_mw = y_test_arr
        y_pred_mw = y_pred_scaled
        print(f"[evaluate_model] DEBUG scale=normalised  "
              f"y_true=[{y_true_mw.min():.4f}, {y_true_mw.max():.4f}]  "
              f"y_pred=[{y_pred_mw.min():.4f}, {y_pred_mw.max():.4f}]")

    rmse    = np.sqrt(mean_squared_error(y_true_mw, y_pred_mw))
    mae     = mean_absolute_error(y_true_mw, y_pred_mw)
    r2      = r2_score(y_true_mw, y_pred_mw)
    metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
    print(f"[evaluate_model] RMSE={rmse:.4f} MW  MAE={mae:.4f} MW  R2={r2:.4f}")
    print(f"[evaluate_model] R2 >= 0.95 -> {'TARGET MET' if r2 >= 0.95 else 'below 0.95'}")
    return metrics, y_pred_scaled


# ── 5. Multi-model comparison ─────────────────────────────────────────────────

def compare_models(X_train, X_test, y_train, y_test, use_search: bool = True):
    """
    Train RF, SVR, XGBoost with TimeSeriesSplit CV; evaluate all; pick best by R².
    """
    rf_model  = train_random_forest(X_train, y_train, use_search=use_search)
    svr_model = train_svr(X_train, y_train, use_search=use_search)
    xgb_model = train_xgboost(X_train, y_train, use_search=use_search)

    rf_metrics,  y_pred_rf  = evaluate_model(rf_model,  X_test, y_test)
    svr_metrics, y_pred_svr = evaluate_model(svr_model, X_test, y_test)
    xgb_metrics, y_pred_xgb = evaluate_model(xgb_model, X_test, y_test)

    print("\n" + "=" * 52)
    print(f"  {'Model':<16} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 52)
    print(f"  {'RandomForest':<16} {rf_metrics['RMSE']:>8.4f} {rf_metrics['MAE']:>8.4f} {rf_metrics['R2']:>8.4f}")
    print(f"  {'SVR':<16} {svr_metrics['RMSE']:>8.4f} {svr_metrics['MAE']:>8.4f} {svr_metrics['R2']:>8.4f}")
    print(f"  {'XGBoost':<16} {xgb_metrics['RMSE']:>8.4f} {xgb_metrics['MAE']:>8.4f} {xgb_metrics['R2']:>8.4f}")
    print("=" * 52)

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

def run_forecasting_pipeline(X_train, X_test, y_train, y_test,
                             use_search: bool = True):
    """
    Train all models on pre-split data (split done in preprocessing).
    CV always uses TimeSeriesSplit to respect temporal ordering.
    """
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
    X_train, X_test, y_train, y_test, scaler, train_df, test_df = \
        preprocess(_os.path.join(_root, "dataset", "DUQ_hourly.csv"))
    best_model, best_metrics, y_test, best_y_pred, X_test, all_metrics, all_preds = \
        run_forecasting_pipeline(X_train, X_test, y_train, y_test, use_search=False)
    print("\nBest model metrics:", best_metrics)
    print("All metrics:", all_metrics)
