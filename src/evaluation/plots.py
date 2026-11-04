"""Visualization and plotting utilities for FlowCast Multi-Modal Forecasting.

Generates actual-vs-predicted curves, confusion matrices, training loss curves,
and three-model benchmark comparison figures (Random Forest vs GRU vs TCN-Attention).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    ax_time.plot(time_steps, y_pred[:t_slice, sensor_id], label="Predicted Volume (TCN-Attn)", color="#ff7f0e", lw=2, linestyle="--")
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
    title: str = "FlowCast Training and Validation Loss Curves",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot training vs validation loss across epochs."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Training Loss (MSE)", color="#2b5c8f", lw=2)
    ax.plot(epochs, val_losses, label="Validation Loss (MSE)", color="#e25d33", lw=2, linestyle="--")

    best_epoch = int(np.argmin(val_losses)) + 1
    best_loss = val_losses[best_epoch - 1]
    ax.scatter(best_epoch, best_loss, color="#d9534f", s=50, zorder=5, label=f"Best Val (Ep {best_epoch}: {best_loss:.4f})")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def plot_three_model_comparison(
    rf_eval: dict,
    gru_eval: dict,
    tcn_eval: dict,
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Generate 3-model benchmark comparison bar charts."""
    fig, (ax_reg, ax_clf) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Regression (MAE & RMSE)
    metrics_reg = ["MAE", "RMSE"]
    x = np.arange(len(metrics_reg))
    width = 0.25

    rf_reg_vals = [rf_eval["regression"]["mae"], rf_eval["regression"]["rmse"]]
    gru_reg_vals = [gru_eval["regression"]["mae"], gru_eval["regression"]["rmse"]]
    tcn_reg_vals = [tcn_eval["regression"]["mae"], tcn_eval["regression"]["rmse"]]

    r1 = ax_reg.bar(x - width, rf_reg_vals, width, label="Random Forest Baseline", color="#90caf9")
    r2 = ax_reg.bar(x, gru_reg_vals, width, label="FlowCast GRU", color="#42a5f5")
    r3 = ax_reg.bar(x + width, tcn_reg_vals, width, label="FlowCast TCN-Attention", color="#1565c0")

    ax_reg.set_title("Forecasting Error Comparison (Lower is Better)", fontsize=11, fontweight="bold")
    ax_reg.set_xticks(x)
    ax_reg.set_xticklabels(metrics_reg, fontsize=10)
    ax_reg.legend(frameon=True)
    ax_reg.bar_label(r1, padding=3, fmt="%.4f", fontsize=8)
    ax_reg.bar_label(r2, padding=3, fmt="%.4f", fontsize=8)
    ax_reg.bar_label(r3, padding=3, fmt="%.4f", fontsize=8)

    # 2. Classification (Macro-F1 & Accuracy)
    metrics_clf = ["Macro-F1", "Accuracy"]
    x2 = np.arange(len(metrics_clf))

    rf_clf_vals = [rf_eval["classification"]["macro_f1"], rf_eval["classification"]["accuracy"]]
    gru_clf_vals = [gru_eval["classification"]["macro_f1"], gru_eval["classification"]["accuracy"]]
    tcn_clf_vals = [tcn_eval["classification"]["macro_f1"], tcn_eval["classification"]["accuracy"]]

    r4 = ax_clf.bar(x2 - width, rf_clf_vals, width, label="Random Forest Baseline", color="#a5d6a7")
    r5 = ax_clf.bar(x2, gru_clf_vals, width, label="FlowCast GRU", color="#66bb6a")
    r6 = ax_clf.bar(x2 + width, tcn_clf_vals, width, label="FlowCast TCN-Attention", color="#2e7d32")

    ax_clf.set_title("Traffic Band Categorization (Higher is Better)", fontsize=11, fontweight="bold")
    ax_clf.set_xticks(x2)
    ax_clf.set_xticklabels(metrics_clf, fontsize=10)
    ax_clf.set_ylim(0, 1.05)
    ax_clf.legend(frameon=True)
    ax_clf.bar_label(r4, padding=3, fmt="%.4f", fontsize=8)
    ax_clf.bar_label(r5, padding=3, fmt="%.4f", fontsize=8)
    ax_clf.bar_label(r6, padding=3, fmt="%.4f", fontsize=8)

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
    """Backward compatibility wrapper for 2-model comparison."""
    fig, (ax_reg, ax_clf) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    metrics_reg = ["MAE", "RMSE"]
    x = np.arange(len(metrics_reg))
    width = 0.35
    rf_reg_vals = [baseline_reg["mae"], baseline_reg["rmse"]]
    gru_reg_vals = [gru_reg["mae"], gru_reg["rmse"]]
    r1 = ax_reg.bar(x - width / 2, rf_reg_vals, width, label="Random Forest", color="#a6cee3")
    r2 = ax_reg.bar(x + width / 2, gru_reg_vals, width, label="FlowCast GRU", color="#1f78b4")
    ax_reg.set_title("Regression Error (Lower is Better)", fontsize=11, fontweight="bold")
    ax_reg.set_xticks(x)
    ax_reg.set_xticklabels(metrics_reg)
    ax_reg.legend(frameon=True)
    ax_reg.bar_label(r1, padding=3, fmt="%.4f", fontsize=8)
    ax_reg.bar_label(r2, padding=3, fmt="%.4f", fontsize=8)

    metrics_clf = ["Macro-F1", "Accuracy"]
    x2 = np.arange(len(metrics_clf))
    rf_clf_vals = [baseline_clf["macro_f1"], baseline_clf["accuracy"]]
    gru_clf_vals = [gru_clf["macro_f1"], gru_clf["accuracy"]]
    r3 = ax_clf.bar(x2 - width / 2, rf_clf_vals, width, label="Random Forest", color="#b2df8a")
    r4 = ax_clf.bar(x2 + width / 2, gru_clf_vals, width, label="FlowCast GRU", color="#33a02c")
    ax_clf.set_title("Classification (Higher is Better)", fontsize=11, fontweight="bold")
    ax_clf.set_xticks(x2)
    ax_clf.set_xticklabels(metrics_clf)
    ax_clf.set_ylim(0, 1.05)
    ax_clf.legend(frameon=True)
    ax_clf.bar_label(r3, padding=3, fmt="%.4f", fontsize=8)
    ax_clf.bar_label(r4, padding=3, fmt="%.4f", fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig
