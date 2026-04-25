"""
src/lstm_model.py
-----------------
PyTorch LSTM for DUQ hourly load forecasting.

Architecture
------------
  Input  : sliding window of WINDOW_SIZE=48 hours × n_features
  Layer 1: LSTM(hidden=128, dropout=0.2)
  Layer 2: LSTM(hidden=64,  dropout=0.2)
  Output : Linear(64 → 1)  — next-hour load in normalised units

Design notes
------------
- Uses the same MinMaxScaler-normalised X_train / X_test produced by
  preprocessing.py, so no separate feature scaling is needed.
- Target (y) is normalised independently using y_train min/max only
  (no leakage) and inverse-transformed before metric reporting.
- Sliding window: each sample is X[i-48 : i] → y[i].
  The first 48 rows of each split are consumed as context, so
  LSTM predictions are aligned to indices [48:] of the original arrays.
- Training uses Adam + ReduceLROnPlateau with early stopping (patience=10).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(_SRC_DIR), "models", "lstm_model.pt")

WINDOW_SIZE = 48      # hours of history fed to the LSTM
HIDDEN1     = 128
HIDDEN2     = 64
DROPOUT     = 0.2
BATCH_SIZE  = 256
MAX_EPOCHS  = 50
LR          = 1e-3
PATIENCE    = 10      # early-stopping patience (epochs)


# ── 1. Dataset ────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Sliding-window dataset.
    X shape : (n_samples, WINDOW_SIZE, n_features)
    y shape : (n_samples,)
    Sample i uses X[i : i+WINDOW_SIZE] to predict y[i+WINDOW_SIZE].
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, window: int = WINDOW_SIZE):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = window

    def __len__(self):
        return len(self.y) - self.w

    def __getitem__(self, idx):
        return self.X[idx : idx + self.w], self.y[idx + self.w]


# ── 2. Model ──────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    Two stacked LSTM layers followed by a linear output head.

    Layer 1: LSTM(n_features → HIDDEN1) with dropout on outputs
    Layer 2: LSTM(HIDDEN1    → HIDDEN2) with dropout on outputs
    Head   : Linear(HIDDEN2  → 1)
    """
    def __init__(self, n_features: int,
                 hidden1: int = HIDDEN1,
                 hidden2: int = HIDDEN2,
                 dropout: float = DROPOUT):
        super().__init__()
        # batch_first=True → input shape (batch, seq, features)
        self.lstm1   = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1   = nn.Dropout(dropout)
        self.lstm2   = nn.LSTM(hidden1,    hidden2, batch_first=True)
        self.drop2   = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden2, 1)

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm1(x)           # (batch, seq, hidden1)
        out     = self.drop1(out)
        out, _ = self.lstm2(out)         # (batch, seq, hidden2)
        out     = self.drop2(out)
        out     = out[:, -1, :]          # take last time-step: (batch, hidden2)
        return self.fc(out).squeeze(-1)  # (batch,)


# ── 3. Target scaler (y only, fit on train) ───────────────────────────────────

class MinMaxTargetScaler:
    """Scales y to [0,1] using train min/max — no leakage."""
    def fit(self, y: np.ndarray):
        self.min_ = y.min()
        self.max_ = y.max()
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.min_) / (self.max_ - self.min_ + 1e-8)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * (self.max_ - self.min_ + 1e-8) + self.min_


# ── 4. Training loop ──────────────────────────────────────────────────────────

def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val:   np.ndarray, y_val:   np.ndarray,
               n_features: int,
               epochs: int    = MAX_EPOCHS,
               lr:     float  = LR,
               device: str    = "cpu") -> tuple:
    """
    Train the LSTM and return (model, y_scaler, train_losses, val_losses).

    X_train / X_val are already MinMaxScaler-normalised feature arrays.
    y is normalised internally using MinMaxTargetScaler fit on y_train only.
    """
    # Scale targets
    y_scaler = MinMaxTargetScaler().fit(y_train)
    y_tr_sc  = y_scaler.transform(y_train)
    y_va_sc  = y_scaler.transform(y_val)

    train_ds = WindowDataset(X_train, y_tr_sc)
    val_ds   = WindowDataset(X_val,   y_va_sc)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = LSTMForecaster(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_ds)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        v_loss /= len(val_ds)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [LSTM] epoch {epoch:>3}/{epochs}  "
                  f"train_loss={t_loss:.6f}  val_loss={v_loss:.6f}")

        # ── early stopping ─────────────────────────────────────────────────
        if v_loss < best_val_loss - 1e-6:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [LSTM] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"  [LSTM] Best val_loss = {best_val_loss:.6f}")
    return model, y_scaler, train_losses, val_losses


# ── 5. Inference ──────────────────────────────────────────────────────────────

def predict_lstm(model: LSTMForecaster,
                 y_scaler,
                 X: np.ndarray,
                 device: str = "cpu") -> np.ndarray:
    """
    Run inference on X and return predictions in original MW units.
    Returns array of length len(X) - WINDOW_SIZE.
    
    y_scaler can be either:
    - MinMaxTargetScaler (internal)
    - sklearn MinMaxScaler (from preprocessing.py)
    """
    model.eval()
    dummy_y = np.zeros(len(X))
    ds      = WindowDataset(X, dummy_y)
    dl      = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds   = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(model(xb.to(device)).cpu().numpy())
    preds_sc = np.concatenate(preds)
    
    # Handle both sklearn and internal scaler
    from src.evaluation import inverse_transform_predictions
    if hasattr(y_scaler, 'inverse_transform') and hasattr(y_scaler, 'data_min_'):
        # sklearn MinMaxScaler
        return inverse_transform_predictions(preds_sc, y_scaler)
    else:
        # Internal MinMaxTargetScaler
        return y_scaler.inverse_transform(preds_sc)


# ── 6. Evaluate ───────────────────────────────────────────────────────────────

def evaluate_lstm(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
    print(f"  [LSTM] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    print(f"  [LSTM] R² >= 0.95 -> {'TARGET MET' if r2 >= 0.95 else 'below 0.95'}")
    return metrics


# ── 7. Save / Load ────────────────────────────────────────────────────────────

def save_lstm(model: LSTMForecaster,
              y_scaler,
              n_features: int,
              path: str = MODEL_PATH) -> None:
    """
    Save LSTM model and scaler info.
    Handles both sklearn MinMaxScaler and internal MinMaxTargetScaler.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Extract min/max from either scaler type
    if hasattr(y_scaler, 'data_min_') and hasattr(y_scaler, 'data_max_'):
        # sklearn MinMaxScaler
        y_min = float(y_scaler.data_min_[0])
        y_max = float(y_scaler.data_max_[0])
    else:
        # Internal MinMaxTargetScaler
        y_min = float(y_scaler.min_)
        y_max = float(y_scaler.max_)
    
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "y_min":      y_min,
        "y_max":      y_max,
        "hidden1":    HIDDEN1,
        "hidden2":    HIDDEN2,
        "dropout":    DROPOUT,
    }, path)
    print(f"  [LSTM] Model saved -> {path}")


def load_lstm(path: str = MODEL_PATH) -> tuple:
    """Returns (model, y_scaler) ready for inference."""
    ckpt     = torch.load(path, map_location="cpu", weights_only=False)
    model    = LSTMForecaster(ckpt["n_features"],
                               ckpt["hidden1"],
                               ckpt["hidden2"],
                               ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    y_scaler      = MinMaxTargetScaler()
    y_scaler.min_ = ckpt["y_min"]
    y_scaler.max_ = ckpt["y_max"]
    print(f"  [LSTM] Model loaded <- {path}")
    return model, y_scaler


# ── 8. Master pipeline ────────────────────────────────────────────────────────

def run_lstm_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                      X_test:  np.ndarray, y_test:  np.ndarray,
                      target_scaler = None,
                      device:  str = "cpu") -> tuple:
    """
    Full LSTM pipeline: train → evaluate → save.

    CRITICAL: To ensure fair comparison with RF/SVR/XGBoost:
    - If target_scaler is provided, y_train and y_test are ALREADY scaled [0,1]
    - LSTM will use the SAME scaler (passed from preprocessing.py)
    - Predictions are inverse-transformed ONCE using the shared scaler
    - All models (RF/SVR/XGB/LSTM) now use identical scaling

    Uses last 20% of X_train as validation (chronological, no shuffle).
    Returns (metrics_dict, y_pred_mw, y_test_mw, train_losses, val_losses).

    y_pred_mw and y_test_mw are both in MW units and trimmed to [WINDOW_SIZE:]
    so they align with the sliding-window offset.
    """
    n_features = X_train.shape[1]

    # Chronological validation split from training data
    val_split  = int(len(X_train) * 0.8)
    X_tr, X_va = X_train[:val_split], X_train[val_split:]
    y_tr, y_va = y_train[:val_split], y_train[val_split:]

    print(f"\n[run_lstm_pipeline] Features={n_features}  "
          f"Train={len(X_tr):,}  Val={len(X_va):,}  "
          f"Test={len(X_test):,}  Window={WINDOW_SIZE}  Device={device}")
    print(f"  Input y_train range: [{y_train.min():.4f}, {y_train.max():.4f}] (scaled)")
    print(f"  Input y_test range:  [{y_test.min():.4f}, {y_test.max():.4f}] (scaled)")
    print(f"  CRITICAL: y_train and y_test are ALREADY scaled [0,1] from preprocessing")
    print(f"  CRITICAL: Will use data as-is, NO additional scaling")

    # Use shared target_scaler if provided, otherwise create internal one
    if target_scaler is not None:
        print("  [LSTM] Using SHARED target_scaler from preprocessing (consistent with RF/XGB/SVR)")
        y_scaler = target_scaler
        # CRITICAL: y_train and y_test are ALREADY scaled [0,1] from preprocessing
        # Do NOT scale again - use directly for training
        y_tr_for_training = y_tr
        y_va_for_training = y_va
    else:
        print("  [LSTM] Creating INTERNAL target_scaler (standalone mode)")
        y_scaler = MinMaxTargetScaler().fit(y_tr)
        y_tr_for_training = y_scaler.transform(y_tr)
        y_va_for_training = y_scaler.transform(y_va)

    # Train LSTM with scaled targets
    train_ds = WindowDataset(X_tr, y_tr_for_training)
    val_ds   = WindowDataset(X_va, y_va_for_training)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model     = LSTMForecaster(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    train_losses, val_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_ds)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        v_loss /= len(val_ds)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [LSTM] epoch {epoch:>3}/{MAX_EPOCHS}  "
                  f"train_loss={t_loss:.6f}  val_loss={v_loss:.6f}")

        # ── early stopping ─────────────────────────────────────────────────
        if v_loss < best_val_loss - 1e-6:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [LSTM] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"  [LSTM] Best val_loss = {best_val_loss:.6f}")

    # Predict on test set — predictions start at index WINDOW_SIZE
    y_pred_mw      = predict_lstm(model, y_scaler, X_test, device=device)
    y_test_aligned = y_test[WINDOW_SIZE:]   # align ground truth to same offset

    # CRITICAL: Inverse transform y_test ONLY ONCE
    # y_test_aligned is scaled [0,1] from preprocessing
    # y_pred_mw is already in MW (predict_lstm calls inverse_transform)
    from src.evaluation import inverse_transform_predictions
    if hasattr(y_scaler, 'inverse_transform') and hasattr(y_scaler, 'data_min_'):
        # sklearn MinMaxScaler from preprocessing
        y_test_mw = inverse_transform_predictions(y_test_aligned, y_scaler)
    else:
        # Internal MinMaxTargetScaler
        y_test_mw = y_scaler.inverse_transform(y_test_aligned)

    print(f"  [LSTM] FINAL y_pred_mw range: [{y_pred_mw.min():.2f}, {y_pred_mw.max():.2f}] MW")
    print(f"  [LSTM] FINAL y_test_mw range: [{y_test_mw.min():.2f}, {y_test_mw.max():.2f}] MW")
    print(f"  [LSTM] Validation: Both arrays in MW units, ready for fair comparison")

    metrics = evaluate_lstm(y_test_mw, y_pred_mw)
    save_lstm(model, y_scaler, n_features)

    return metrics, y_pred_mw, y_test_mw, train_losses, val_losses


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(_SRC_DIR))
    from src.preprocessing import preprocess
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    X_train, X_test, y_train, y_test, scaler, train_df, test_df = \
        preprocess(_os.path.join(_root, "dataset", "DUQ_hourly.csv"))
    metrics, y_pred, y_true, tl, vl = run_lstm_pipeline(
        X_train.values, y_train.values,
        X_test.values,  y_test.values,
    )
    print("LSTM metrics:", metrics)
