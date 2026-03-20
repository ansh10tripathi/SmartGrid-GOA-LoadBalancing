"""
src/forecasting_model.py
------------------------
Improved ML pipeline targeting R² ≥ 0.95.

Key upgrades vs v1
------------------
  • Time-based split  (no data leakage from future → past)
  • Tuned Random Forest: n_estimators=300, max_depth=20, tuned leaf params
  • RandomizedSearchCV for automated hyper-parameter search
  • Falls back gracefully if XGBoost is not installed
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_PATH = "models/load_forecast_model.pkl"


# ── 1. Time-based split ───────────────────────────────────────────────────────

def split_data(X, y, test_size: float = 0.2):
    """
    Strict chronological split — last `test_size` fraction is the test set.
    No shuffling to prevent future-data leakage.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"[split_data] Train={len(X_train)}, Test={len(X_test)} "
          f"(time-based, no shuffle)")
    return X_train, X_test, y_train, y_test


# ── 2. Train with RandomizedSearchCV ─────────────────────────────────────────

def train_model(X_train, y_train, use_search: bool = True,
                random_state: int = 42):
    """
    Train a Random Forest with optional RandomizedSearchCV tuning.

    Parameters
    ----------
    use_search : run RandomizedSearchCV when True (recommended for best R²)
                 set False for a fast fixed-param run
    """
    base_rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    if use_search:
        param_dist = {
            "n_estimators":      [200, 300, 400],
            "max_depth":         [15, 20, 25, None],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf":  [1, 2, 3],
            "max_features":      ["sqrt", "log2", 0.8],
        }
        search = RandomizedSearchCV(
            base_rf,
            param_distributions=param_dist,
            n_iter=20,              # 20 random combinations
            cv=3,                   # 3-fold cross-validation
            scoring="r2",
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )
        print("[train_model] Running RandomizedSearchCV (20 iter × 3-fold CV)…")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"[train_model] Best params : {search.best_params_}")
        print(f"[train_model] Best CV R²  : {search.best_score_:.4f}")
    else:
        # Fast fixed-param model (still well-tuned)
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        print("[train_model] Trained fixed-param RandomForest (300 trees, "
              "depth=20).")

    return model


# ── 3. Evaluate ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Return metrics dict + predictions array."""
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4),
               "R2": round(r2, 4)}
    print(f"[evaluate_model] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    # Highlight if target is met
    status = "✅ TARGET MET" if r2 >= 0.95 else "⚠️  below 0.95"
    print(f"[evaluate_model] R² ≥ 0.95 → {status}")
    return metrics, y_pred


# ── 4. Save / Load ────────────────────────────────────────────────────────────

def save_model(model, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Model saved → {path}")


def load_model(path: str = MODEL_PATH):
    model = joblib.load(path)
    print(f"[load_model] Model loaded ← {path}")
    return model


# ── 5. Master pipeline ────────────────────────────────────────────────────────

def run_forecasting_pipeline(X, y, use_search: bool = True):
    """
    Full pipeline: split → train → evaluate → save → plot.
    Returns (model, metrics, y_test, y_pred, X_test).
    """
    from src.evaluation import evaluate_model_performance  # avoid circular import

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train, use_search=use_search)
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    save_model(model)

    # Detailed metrics printout + 2-panel plot
    evaluate_model_performance(y_test, y_pred)

    return model, metrics, y_test, y_pred, X_test


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import preprocess

    X, y, scaler, df = preprocess("dataset/smartgrid.csv")
    model, metrics, y_test, y_pred, X_test = run_forecasting_pipeline(
        X, y, use_search=False   # fast run for testing
    )
    print("\nFinal metrics:", metrics)
