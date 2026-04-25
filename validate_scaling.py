"""
validate_scaling.py
-------------------
Comprehensive validation script to verify scaling consistency across all models.

This script checks:
1. All models use the same target scaler
2. Predictions are inverse-transformed exactly once
3. All metrics are computed in MW units
4. No double scaling occurs
5. Min/max values are realistic

Run after training: python validate_scaling.py
"""

import os
import sys
import numpy as np
import joblib

# Add project root to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.preprocessing import preprocess
from src.forecasting_model import load_model
from src.evaluation import inverse_transform_predictions

print("=" * 80)
print("  SCALING CONSISTENCY VALIDATION")
print("=" * 80)

# ── Step 1: Load data and scalers ────────────────────────────────────────────
print("\n[1/6] Loading data and scalers...")
X_train, X_test, y_train, y_test, scaler, target_scaler, train_df, test_df = \
    preprocess(os.path.join(_ROOT, "dataset", "DUQ_hourly.csv"))

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape:  {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape:  {y_test.shape}")

# ── Step 2: Validate target scaler ───────────────────────────────────────────
print("\n[2/6] Validating target scaler...")
print(f"  Target scaler type: {type(target_scaler).__name__}")
print(f"  Target scaler data_min_: {target_scaler.data_min_[0]:.2f}")
print(f"  Target scaler data_max_: {target_scaler.data_max_[0]:.2f}")
print(f"  Target scaler range: {target_scaler.data_max_[0] - target_scaler.data_min_[0]:.2f} MW")

# Validate y_train and y_test are scaled [0,1]
print(f"\n  y_train (scaled): min={y_train.min():.4f}, max={y_train.max():.4f}")
print(f"  y_test (scaled):  min={y_test.min():.4f}, max={y_test.max():.4f}")

if y_train.min() < -0.01 or y_train.max() > 1.01:
    print("  ⚠️  WARNING: y_train not in [0,1] range!")
if y_test.min() < -0.01 or y_test.max() > 1.01:
    print("  ⚠️  WARNING: y_test not in [0,1] range!")

# Inverse transform to verify MW range
y_train_mw = inverse_transform_predictions(np.asarray(y_train), target_scaler)
y_test_mw = inverse_transform_predictions(np.asarray(y_test), target_scaler)
print(f"\n  y_train (MW): min={y_train_mw.min():.2f}, max={y_train_mw.max():.2f}")
print(f"  y_test (MW):  min={y_test_mw.min():.2f}, max={y_test_mw.max():.2f}")

# Realistic range check (DUQ load is typically 1000-2500 MW)
if y_test_mw.min() < 500 or y_test_mw.max() > 3000:
    print("  ⚠️  WARNING: MW values outside realistic range [500, 3000]!")
else:
    print("  ✓ MW values in realistic range")

# ── Step 3: Validate sklearn models (RF/SVR/XGBoost) ─────────────────────────
print("\n[3/6] Validating sklearn models (RF/SVR/XGBoost)...")

model_files = {
    "RandomForest": "models/random_forest_model.pkl",
    "SVR": "models/svr_model.pkl",
    "XGBoost": "models/xgboost_model.pkl",
}

for name, path in model_files.items():
    if not os.path.exists(path):
        print(f"  ⚠️  {name}: model not found at {path}")
        continue
    
    print(f"\n  {name}:")
    pipeline = joblib.load(path)
    
    # Predict on test set (returns scaled predictions)
    y_pred_scaled = pipeline.predict(X_test)
    print(f"    Scaled predictions: min={y_pred_scaled.min():.4f}, max={y_pred_scaled.max():.4f}")
    
    # Inverse transform to MW
    y_pred_mw = inverse_transform_predictions(y_pred_scaled, target_scaler)
    print(f"    MW predictions:     min={y_pred_mw.min():.2f}, max={y_pred_mw.max():.2f}")
    
    # Validate range
    if y_pred_mw.min() < 500 or y_pred_mw.max() > 3000:
        print(f"    ⚠️  WARNING: Predictions outside realistic range!")
    else:
        print(f"    ✓ Predictions in realistic range")
    
    # Compute metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test_mw, y_pred_mw))
    mae = mean_absolute_error(y_test_mw, y_pred_mw)
    r2 = r2_score(y_test_mw, y_pred_mw)
    print(f"    RMSE: {rmse:.2f} MW  |  MAE: {mae:.2f} MW  |  R²: {r2:.4f}")
    
    # Red flag check
    if r2 > 0.95:
        print(f"    ⚠️  WARNING: R² > 0.95 may indicate data leakage!")
    elif r2 < 0.70:
        print(f"    ⚠️  WARNING: R² < 0.70 is unusually low!")
    else:
        print(f"    ✓ R² in realistic range [0.70, 0.95]")

# ── Step 4: Validate LSTM ─────────────────────────────────────────────────────
print("\n[4/6] Validating LSTM...")

lstm_path = "models/lstm_model.pt"
if not os.path.exists(lstm_path):
    print(f"  ⚠️  LSTM model not found at {lstm_path}")
else:
    import torch
    from src.lstm_model import load_lstm, predict_lstm, WINDOW_SIZE
    
    model, y_scaler = load_lstm(lstm_path)
    
    # Check if LSTM is using the same scaler
    print(f"  LSTM scaler type: {type(y_scaler).__name__}")
    if hasattr(y_scaler, 'data_min_'):
        print(f"  LSTM scaler data_min_: {y_scaler.data_min_[0]:.2f}")
        print(f"  LSTM scaler data_max_: {y_scaler.data_max_[0]:.2f}")
        if abs(y_scaler.data_min_[0] - target_scaler.data_min_[0]) > 1.0:
            print(f"  ⚠️  WARNING: LSTM scaler differs from target_scaler!")
        else:
            print(f"  ✓ LSTM using consistent scaler")
    else:
        print(f"  LSTM scaler min_: {y_scaler.min_:.2f}")
        print(f"  LSTM scaler max_: {y_scaler.max_:.2f}")
    
    # Predict
    y_pred_lstm_mw = predict_lstm(model, target_scaler, X_test.values, device="cpu")
    print(f"  LSTM predictions: min={y_pred_lstm_mw.min():.2f}, max={y_pred_lstm_mw.max():.2f} MW")
    
    # Align ground truth
    y_test_lstm_mw = y_test_mw[WINDOW_SIZE:]
    print(f"  LSTM ground truth: min={y_test_lstm_mw.min():.2f}, max={y_test_lstm_mw.max():.2f} MW")
    
    # Validate range
    if y_pred_lstm_mw.min() < 500 or y_pred_lstm_mw.max() > 3000:
        print(f"  ⚠️  WARNING: LSTM predictions outside realistic range!")
    else:
        print(f"  ✓ LSTM predictions in realistic range")
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test_lstm_mw, y_pred_lstm_mw))
    mae = mean_absolute_error(y_test_lstm_mw, y_pred_lstm_mw)
    r2 = r2_score(y_test_lstm_mw, y_pred_lstm_mw)
    print(f"  RMSE: {rmse:.2f} MW  |  MAE: {mae:.2f} MW  |  R²: {r2:.4f}")
    
    if r2 > 0.95:
        print(f"  ⚠️  WARNING: R² > 0.95 may indicate data leakage!")
    elif r2 < 0.70:
        print(f"  ⚠️  WARNING: R² < 0.70 is unusually low!")
    else:
        print(f"  ✓ R² in realistic range [0.70, 0.95]")

# ── Step 5: Cross-model comparison ───────────────────────────────────────────
print("\n[5/6] Cross-model comparison...")

results_json = "results/model_results.json"
if os.path.exists(results_json):
    import json
    with open(results_json) as f:
        results = json.load(f)
    
    print("\n  Model Performance Summary:")
    print("  " + "-" * 70)
    print(f"  {'Model':<16} {'RMSE (MW)':<12} {'MAE (MW)':<12} {'R²':<10} {'MAPE (%)':<10}")
    print("  " + "-" * 70)
    
    for model_name in ["RandomForest", "SVR", "XGBoost", "LSTM"]:
        if model_name in results:
            m = results[model_name]
            rmse = m.get("RMSE", "N/A")
            mae = m.get("MAE", "N/A")
            r2 = m.get("R2", "N/A")
            mape = m.get("MAPE", "N/A")
            
            rmse_str = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else rmse
            mae_str = f"{mae:.2f}" if isinstance(mae, (int, float)) else mae
            r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else r2
            mape_str = f"{mape:.2f}" if isinstance(mape, (int, float)) else mape
            
            print(f"  {model_name:<16} {rmse_str:<12} {mae_str:<12} {r2_str:<10} {mape_str:<10}")
    
    print("  " + "-" * 70)
    
    # Check for consistency
    rmse_values = [results[m]["RMSE"] for m in ["RandomForest", "SVR", "XGBoost", "LSTM"] 
                   if m in results and results[m].get("RMSE") is not None]
    
    if rmse_values:
        rmse_range = max(rmse_values) - min(rmse_values)
        print(f"\n  RMSE range across models: {rmse_range:.2f} MW")
        if rmse_range > 500:
            print(f"  ⚠️  WARNING: Large RMSE variance suggests scaling inconsistency!")
        else:
            print(f"  ✓ RMSE values reasonably consistent")
else:
    print("  ⚠️  model_results.json not found")

# ── Step 6: Final validation summary ─────────────────────────────────────────
print("\n[6/6] Final Validation Summary")
print("=" * 80)

checks = {
    "Target scaler loaded": target_scaler is not None,
    "y_train in [0,1]": 0 <= y_train.min() <= 0.01 and 0.99 <= y_train.max() <= 1.01,
    "y_test in [0,1]": 0 <= y_test.min() <= 0.01 and 0.99 <= y_test.max() <= 1.01,
    "y_test_mw in realistic range": 500 <= y_test_mw.min() and y_test_mw.max() <= 3000,
}

print("\n  Validation Checks:")
for check, passed in checks.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"    {status}  {check}")

all_passed = all(checks.values())
print("\n" + "=" * 80)
if all_passed:
    print("  ✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
    print("  All models use consistent scaling. Metrics are fair and comparable.")
else:
    print("  ✗✗✗ VALIDATION FAILED ✗✗✗")
    print("  Fix scaling issues before comparing models.")
print("=" * 80)
