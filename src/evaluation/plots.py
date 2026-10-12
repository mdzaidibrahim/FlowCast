"""Visualization and plotting utilities for FlowCast.

Generates actual-vs-predicted curves, confusion matrices, training loss curves,
and baseline comparison figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensor_id: int = 0,
    time_window: int = 150,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot actual vs predicted traffic volume for a specific sensor over time and as a scatter."""
    fig, (ax_time, ax_scatter) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Time series slice
    t_slice = min(time_window, len(y_true))
    time_steps = np.arange(t_slice)
    ax_time.plot(time_steps, y_true[:t_slice, sensor_id], label="Actual Volume", color="#1f77b4", lw=2)
    ax_time.plot(time_steps, y_pred[:t_slice, sensor_id], label="Predicted Volume", color="#ff7f0e", lw=2, linestyle="--")
    ax_time.set_title(f"Next-15m Forecast vs Actual (Sensor #{sensor_id:02d})", fontsize=12, fontweight="bold")
    ax_time.set_xlabel("Time Step (15-min Intervals)", fontsize=10)
    ax_time.set_ylabel("Normalized Traffic Volume", fontsize=10)
    ax_time.legend(frameon=True)
    ax_time.set_ylim(-0.02, 1.02)

    # Scatter with identity line
    s_true = y_true[:, sensor_id]
    s_pred = y_pred[:, sensor_id]
    ax_scatter.scatter(s_true, s_pred, alpha=0.35, color="#2ca02c", s=18, edgecolors="none")
    ideal_line = np.linspace(0, 1, 100)
    ax_scatter.plot(ideal_line, ideal_line, color="#d62728", lw=1.5, linestyle=":", label="Ideal (y = x)")
    ax_scatter.set_title(f"Prediction Fidelity (Sensor #{sensor_id:02d})", fontsize=12, fontweight="bold")
    ax_scatter.set_xlabel("Actual Volume", fontsize=10)
    ax_scatter.set_ylabel("Predicted Volume", fontsize=10)
    ax_scatter.legend(frameon=True)
    ax_scatter.set_xlim(-0.02, 1.02)
    ax_scatter.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Traffic Volume Band Confusion Matrix",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot an annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Traffic Band", fontsize=10)
    ax.set_ylabel("Actual Traffic Band", fontsize=10)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "FlowCast GRU: Training vs Validation Loss (MSE)",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot training and validation loss curves over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss (MSE)", color="#1f77b4", lw=2)
    ax.plot(epochs, val_losses, label="Validation Loss (MSE)", color="#d62728", lw=2, linestyle="--")

    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    ax.scatter([best_epoch], [best_val], color="#2ca02c", s=60, zorder=5, label=f"Best Checkpoint (Ep {best_epoch}: {best_val:.4f})")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_model_comparison(
    baseline_reg: dict,
    gru_reg: dict,
    baseline_clf: dict,
    gru_clf: dict,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Generate side-by-side metric comparison bar chart between Baseline and GRU."""
    fig, (ax_reg, ax_clf) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Regression comparison (MAE & RMSE - lower is better)
    metrics_reg = ["MAE", "RMSE"]
    x = np.arange(len(metrics_reg))
    width = 0.35

    rf_reg_vals = [baseline_reg["mae"], baseline_reg["rmse"]]
    gru_reg_vals = [gru_reg["mae"], gru_reg["rmse"]]

    rects1 = ax_reg.bar(x - width / 2, rf_reg_vals, width, label="Random Forest", color="#a6cee3")
    rects2 = ax_reg.bar(x + width / 2, gru_reg_vals, width, label="FlowCast GRU", color="#1f78b4")

    ax_reg.set_title("Regression Error (Lower is Better)", fontsize=11, fontweight="bold")
    ax_reg.set_xticks(x)
    ax_reg.set_xticklabels(metrics_reg)
    ax_reg.legend(frameon=True)
    ax_reg.bar_label(rects1, padding=3, fmt="%.4f", fontsize=8)
    ax_reg.bar_label(rects2, padding=3, fmt="%.4f", fontsize=8)

    # Classification comparison (Macro-F1 & Accuracy - higher is better)
    metrics_clf = ["Macro-F1", "Accuracy"]
    x2 = np.arange(len(metrics_clf))

    rf_clf_vals = [baseline_clf["macro_f1"], baseline_clf["accuracy"]]
    gru_clf_vals = [gru_clf["macro_f1"], gru_clf["accuracy"]]

    rects3 = ax_clf.bar(x2 - width / 2, rf_clf_vals, width, label="Random Forest", color="#b2df8a")
    rects4 = ax_clf.bar(x2 + width / 2, gru_clf_vals, width, label="FlowCast GRU", color="#33a02c")

    ax_clf.set_title("Traffic Band Classification (Higher is Better)", fontsize=11, fontweight="bold")
    ax_clf.set_xticks(x2)
    ax_clf.set_xticklabels(metrics_clf)
    ax_clf.set_ylim(0, 1.05)
    ax_clf.legend(frameon=True)
    ax_clf.bar_label(rects3, padding=3, fmt="%.4f", fontsize=8)
    ax_clf.bar_label(rects4, padding=3, fmt="%.4f", fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig
