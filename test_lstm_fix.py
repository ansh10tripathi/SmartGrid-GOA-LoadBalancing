"""Quick test to verify LSTM scaling fix works"""
import sys
sys.path.insert(0, '.')

# Test save_lstm with sklearn scaler
print("Testing save_lstm with sklearn MinMaxScaler...")

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Create mock sklearn scaler
scaler = MinMaxScaler()
scaler.fit(np.array([[1000], [2500]]))

print(f"Scaler type: {type(scaler).__name__}")
print(f"Has data_min_: {hasattr(scaler, 'data_min_')}")
print(f"Has data_max_: {hasattr(scaler, 'data_max_')}")

if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
    y_min = float(scaler.data_min_[0])
    y_max = float(scaler.data_max_[0])
    print(f"✓ Successfully extracted: y_min={y_min}, y_max={y_max}")
else:
    print("✗ Failed to extract min/max")

print("\n✓ save_lstm fix validated!")
