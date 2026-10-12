"""Evaluation metrics module for FlowCast.

Computes continuous regression metrics (MAE, RMSE, R2) and derived discrete
traffic-volume band classification metrics (Precision, Recall, Macro-F1, Confusion Matrix).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from ..data.preprocess import classify_traffic


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute network-wide MAE, RMSE, and R2 regression metrics.

    Args:
        y_true: Ground truth traffic volumes, shape (N, 36) or (N,).
        y_pred: Predicted traffic volumes, shape (N, 36) or (N,).

    Returns:
        Dict with keys: 'mae', 'rmse', 'r2'.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true.flatten(), y_pred.flatten()))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_per_sensor_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute MAE, RMSE, and R2 individually for each of the 36 sensor stations.

    Args:
        y_true: Array of shape (N, 36).
        y_pred: Array of shape (N, 36).

    Returns:
        Dict mapping sensor_id (0..35) to {'mae': float, 'rmse': float, 'r2': float}.
    """
    num_sensors = y_true.shape[1]
    per_sensor = {}
    for s in range(num_sensors):
        s_true = y_true[:, s]
        s_pred = y_pred[:, s]
        mae = float(mean_absolute_error(s_true, s_pred))
        rmse = float(np.sqrt(mean_squared_error(s_true, s_pred)))
        r2 = float(r2_score(s_true, s_pred))
        per_sensor[s] = {"mae": mae, "rmse": rmse, "r2": r2}
    return per_sensor


def compute_classification_metrics(
    y_true_cont: np.ndarray,
    y_pred_cont: np.ndarray,
    thresholds: Tuple[float, float],
) -> Dict[str, Any]:
    """Derive traffic-volume bands and calculate classification metrics.

    Args:
        y_true_cont: Ground truth continuous traffic volume array.
        y_pred_cont: Predicted continuous traffic volume array.
        thresholds: (low_threshold, high_threshold) derived strictly from training data.

    Returns:
        Dict with precision, recall, macro_f1, accuracy, and confusion_matrix.
    """
    y_true_cat = classify_traffic(y_true_cont.flatten(), thresholds)
    y_pred_cat = classify_traffic(y_pred_cont.flatten(), thresholds)

    labels = [0, 1, 2]
    precision = float(
        precision_score(
            y_true_cat, y_pred_cat, labels=labels, average="macro", zero_division=0
        )
    )
    recall = float(
        recall_score(
            y_true_cat, y_pred_cat, labels=labels, average="macro", zero_division=0
        )
    )
    macro_f1 = float(
        f1_score(
            y_true_cat, y_pred_cat, labels=labels, average="macro", zero_division=0
        )
    )
    acc = float(accuracy_score(y_true_cat, y_pred_cat))
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)

    return {
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "confusion_matrix": cm,
        "class_names": ["Low Volume", "Moderate Volume", "High Volume"],
    }


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Tuple[float, float],
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Comprehensive evaluation returning both regression and classification metrics."""
    reg = compute_regression_metrics(y_true, y_pred)
    clf = compute_classification_metrics(y_true, y_pred, thresholds)
    return {
        "model_name": model_name,
        "regression": reg,
        "classification": clf,
    }
