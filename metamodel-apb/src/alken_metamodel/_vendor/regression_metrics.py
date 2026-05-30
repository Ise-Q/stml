"""Regression evaluation metrics with a zero-safe MAPE.

Source: T3.03_PS4_Solutions.ipynb (Madmoun, L4 regression losses). MAPE is
scale-free but undefined where y == 0, so those entries are masked before
dividing. Use one helper so every model is scored identically.

Stack: numpy, pandas, scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%), skipping y_true == 0 entries."""
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    mask = np.asarray(y_true) != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true, y_pred, model_name):
    """Return a dict with MAE, MSE, RMSE, MAPE (%), and R^2."""
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    mse = mean_squared_error(y_true, y_pred)
    return {
        'Model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAPE (%)': calculate_mape(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
    }


def report_metrics(metrics_dict):
    """Print a formatted metrics report."""
    print(f"\n{metrics_dict['Model']} Results:")
    print(f"  MAE: {metrics_dict['MAE']:.2f}")
    print(f"  MSE: {metrics_dict['MSE']:.2f}")
    print(f"  RMSE: {metrics_dict['RMSE']:.2f}")
    print(f"  MAPE: {metrics_dict['MAPE (%)']:.2f}%")
    print(f"  R²: {metrics_dict['R²']:.4f}")


if __name__ == "__main__":
    y = np.array([0.0, 1.0, 2.0, 4.0, 8.0])      # includes a zero
    yhat = np.array([0.1, 1.1, 1.8, 4.2, 7.5])
    report_metrics(calculate_metrics(y, yhat, "demo"))   # MAPE ignores the zero row
