import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Load arrays ----
y_true = np.load('/Users/rajaryan/Projects/ML/SDoH-ER-Predictor/data/processed/y_test.npy')  # Adjust path as needed
y_pred = np.load('/Users/rajaryan/Projects/ML/SDoH-ER-Predictor/backend/ml/y_pred.npy')  # Adjust path as needed

# ---- Flatten if needed ----
y_true = y_true.flatten()
y_pred = y_pred.flatten()

# ---- Standard Metrics ----
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")
