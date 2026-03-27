
import numpy as np
import pickle

# ── Paths — update to your actual paths ───────────────────────────────────────
PROCESSED_DIR = "/Users/rajaryan/Projects/ML/SDoH-ER-Predictor/data/processed/"
xgb_model_path = PROCESSED_DIR + "xgb_model.pkl"          # NEW — was er_lstm_model.h5
scaler_path    = PROCESSED_DIR + "scaler.pkl"
x_test_path    = PROCESSED_DIR + "X_test.npy"

# ── Load scaler (unchanged) ────────────────────────────────────────────────────
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# ── Load XGBoost model (replaces Keras load_model) ────────────────────────────
with open(xgb_model_path, "rb") as f:
    model = pickle.load(f)

# ── Load X_test (unchanged) ───────────────────────────────────────────────────
X_test = np.load(x_test_path)

# ── Scale X_test (unchanged logic) ────────────────────────────────────────────
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# ── Flatten to 2D for XGBoost — only change from original ─────────────────────
X_test_2d = X_test_scaled.reshape(X_test_scaled.shape[0], -1)

# ── Predict ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_2d)

# ── Save (unchanged) ──────────────────────────────────────────────────────────
np.save(PROCESSED_DIR + "y_pred.npy", y_pred)
print("Prediction complete.")
print("Shape:", y_pred.shape)
print("Sample predictions:\n", y_pred[:15])