"""End-to-end training and evaluation pipeline for FlowCast.

Trains:
1. Baseline Model (Multi-Output Random Forest)
2. Spatio-Temporal GRU Network
3. Deep Dilated TCN with Multi-Head Temporal Self-Attention (Novel Architecture)

Logs training loss curves, evaluates comprehensive regression & multi-class metrics,
calculates multi-modal indicators (CO2 & EV surge), and serializes all evaluation artifacts.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.multimodal import (
    compute_carbon_emissions,
    compute_ev_grid_surge_index,
    compute_green_wave_advisory,
)
from ..data.preprocess import prepare_data_splits
from ..evaluation.metrics import evaluate_model_performance
from ..evaluation.plots import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_three_model_comparison,
    plot_training_curves,
)
from .baseline import FlowCastBaseline
from .gru_model import FlowCastGRU, get_default_device
from .tcn_model import FlowCastTCNAttention, save_tcn_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FlowCast")


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set PyTorch CPU threads safely on Windows to prevent OpenMP stack overflow
    try:
        torch.set_num_threads(2)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_gru(
    splits,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.002,
    patience: int = 10,
    checkpoint_path: Path = Path("artifacts/models/gru_best.pth"),
) -> Tuple[FlowCastGRU, list[float], list[float]]:
    """Train FlowCastGRU with early stopping and learning rate scheduling."""
    logger.info("Initializing FlowCastGRU on device: %s", device)
    model = FlowCastGRU(
        num_sensors=36,
        seq_len=10,
        hidden_dim=64,
        num_layers=2,
        context_dim=32,
        dropout=0.15,
    ).to(device)

    train_dataset = TensorDataset(
        torch.tensor(splits.X_train_seq, dtype=torch.float32),
        torch.tensor(splits.X_train_ctx, dtype=torch.float32),
        torch.tensor(splits.Y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_seq = torch.tensor(splits.X_val_seq, dtype=torch.float32).to(device)
    val_ctx = torch.tensor(splits.X_val_ctx, dtype=torch.float32).to(device)
    val_y = torch.tensor(splits.Y_val, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for b_seq, b_ctx, b_y in train_loader:
            b_seq = b_seq.to(device)
            b_ctx = b_ctx.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            preds = model(b_seq, b_ctx)
            loss = criterion(preds, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * len(b_y)

        train_loss = running_train_loss / len(train_dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_preds = model(val_seq, val_ctx)
            val_loss = criterion(val_preds, val_y).item()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(checkpoint_path)
            status_tag = "*"
        else:
            patience_counter += 1
            status_tag = " "

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            logger.info(
                "GRU Epoch [%02d/%02d] | Train Loss: %.5f | Val Loss: %.5f %s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                status_tag,
            )

        if patience_counter >= patience:
            logger.info("GRU Early stopping at epoch %d. Best Val: %.5f", epoch, best_val_loss)
            break

    best_model = FlowCastGRU.load(checkpoint_path, device=device)
    return best_model, train_losses, val_losses


def train_tcn_attention(
    splits,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.002,
    patience: int = 10,
    checkpoint_path: Path = Path("artifacts/models/tcn_attention_best.pth"),
) -> Tuple[FlowCastTCNAttention, list[float], list[float]]:
    """Train FlowCastTCNAttention with multi-head attention and causal convolutions."""
    logger.info("Initializing FlowCastTCNAttention on device: %s", device)
    model = FlowCastTCNAttention(
        num_sensors=36,
        seq_len=10,
        tcn_channels=(64, 64),
        kernel_size=3,
        num_heads=4,
        context_dim=32,
        dropout=0.15,
    ).to(device)

    train_dataset = TensorDataset(
        torch.tensor(splits.X_train_seq, dtype=torch.float32),
        torch.tensor(splits.X_train_ctx, dtype=torch.float32),
        torch.tensor(splits.Y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_seq = torch.tensor(splits.X_val_seq, dtype=torch.float32).to(device)
    val_ctx = torch.tensor(splits.X_val_ctx, dtype=torch.float32).to(device)
    val_y = torch.tensor(splits.Y_val, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for b_seq, b_ctx, b_y in train_loader:
            b_seq = b_seq.to(device)
            b_ctx = b_ctx.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            preds = model(b_seq, b_ctx)
            loss = criterion(preds, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * len(b_y)

        train_loss = running_train_loss / len(train_dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_preds = model(val_seq, val_ctx)
            val_loss = criterion(val_preds, val_y).item()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_tcn_checkpoint(
                model,
                checkpoint_path,
                hyperparameters={
                    "num_sensors": 36,
                    "seq_len": 10,
                    "tcn_channels": [64, 64],
                    "kernel_size": 3,
                    "num_heads": 4,
                    "context_dim": 32,
                    "dropout": 0.15,
                },
            )
            status_tag = "*"
        else:
            patience_counter += 1
            status_tag = " "

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            logger.info(
                "TCN-Attn Epoch [%02d/%02d] | Train Loss: %.5f | Val Loss: %.5f %s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                status_tag,
            )

        if patience_counter >= patience:
            logger.info("TCN-Attn Early stopping at epoch %d. Best Val: %.5f", epoch, best_val_loss)
            break

    from .tcn_model import load_tcn_checkpoint
    best_model = load_tcn_checkpoint(checkpoint_path, device=device)
    return best_model, train_losses, val_losses


def run_pipeline() -> Dict[str, Any]:
    """Execute complete data loading, model training, evaluation, and artifact generation."""
    set_seed(42)
    device = get_default_device()
    logger.info("Starting FlowCast Multi-Modal training pipeline on device: %s", device)

    # 1. Prepare data splits
    splits = prepare_data_splits(val_size=200)
    logger.info(
        "Data splits ready. Train: %d, Val: %d, Test: %d timestamps.",
        len(splits.X_train_seq),
        len(splits.X_val_seq),
        len(splits.X_test_seq),
    )
    logger.info("Classification thresholds (derived from train): %s", splits.thresholds)

    # 2. Train baseline (Random Forest)
    logger.info("Training Random Forest baseline...")
    rf = FlowCastBaseline(n_estimators=100, max_depth=16, random_state=42)
    t0 = time.time()
    rf.fit(splits.X_train_flat, splits.Y_train)
    rf_train_time = time.time() - t0
    rf_path = Path("artifacts/models/baseline_rf.joblib")
    rf.save(rf_path)
    logger.info("Baseline saved to %s (training time: %.2fs)", rf_path, rf_train_time)

    # 3. Train FlowCastGRU
    gru_path = Path("artifacts/models/gru_best.pth")
    t0 = time.time()
    gru_model, gru_train_losses, gru_val_losses = train_gru(
        splits, device=device, epochs=50, batch_size=32, lr=0.002, checkpoint_path=gru_path
    )
    gru_train_time = time.time() - t0
    logger.info("GRU training finished in %.2fs. Checkpoint at %s", gru_train_time, gru_path)

    # 4. Train FlowCastTCNAttention (Novel architecture)
    tcn_path = Path("artifacts/models/tcn_attention_best.pth")
    t0 = time.time()
    tcn_model, tcn_train_losses, tcn_val_losses = train_tcn_attention(
        splits, device=device, epochs=50, batch_size=32, lr=0.002, checkpoint_path=tcn_path
    )
    tcn_train_time = time.time() - t0
    logger.info("TCN-Attention training finished in %.2fs. Checkpoint at %s", tcn_train_time, tcn_path)

    # 5. Generate Predictions on Held-Out Test Set
    rf_test_preds = rf.predict(splits.X_test_flat)

    test_seq = torch.tensor(splits.X_test_seq, dtype=torch.float32).to(device)
    test_ctx = torch.tensor(splits.X_test_ctx, dtype=torch.float32).to(device)

    gru_model.eval()
    with torch.no_grad():
        gru_test_preds = gru_model(test_seq, test_ctx).cpu().numpy()

    tcn_model.eval()
    with torch.no_grad():
        tcn_test_preds = tcn_model(test_seq, test_ctx).cpu().numpy()

    # 6. Evaluate Metrics for All 3 Models
    rf_eval = evaluate_model_performance(
        splits.Y_test, rf_test_preds, splits.thresholds, model_name="Random Forest"
    )
    gru_eval = evaluate_model_performance(
        splits.Y_test, gru_test_preds, splits.thresholds, model_name="FlowCast GRU"
    )
    tcn_eval = evaluate_model_performance(
        splits.Y_test, tcn_test_preds, splits.thresholds, model_name="FlowCast TCN-Attention"
    )

    logger.info("=== HELD-OUT TEST BENCHMARK COMPARISON ===")
    logger.info(
        "Random Forest       -> MAE: %.4f | RMSE: %.4f | R2: %.4f | Macro-F1: %.4f | Acc: %.2f%%",
        rf_eval["regression"]["mae"],
        rf_eval["regression"]["rmse"],
        rf_eval["regression"]["r2"],
        rf_eval["classification"]["macro_f1"],
        rf_eval["classification"]["accuracy"] * 100,
    )
    logger.info(
        "FlowCast GRU        -> MAE: %.4f | RMSE: %.4f | R2: %.4f | Macro-F1: %.4f | Acc: %.2f%%",
        gru_eval["regression"]["mae"],
        gru_eval["regression"]["rmse"],
        gru_eval["regression"]["r2"],
        gru_eval["classification"]["macro_f1"],
        gru_eval["classification"]["accuracy"] * 100,
    )
    logger.info(
        "FlowCast TCN-Attn   -> MAE: %.4f | RMSE: %.4f | R2: %.4f | Macro-F1: %.4f | Acc: %.2f%%",
        tcn_eval["regression"]["mae"],
        tcn_eval["regression"]["rmse"],
        tcn_eval["regression"]["r2"],
        tcn_eval["classification"]["macro_f1"],
        tcn_eval["classification"]["accuracy"] * 100,
    )

    # 7. Save Metrics Summary JSON
    metrics_path = Path("artifacts/metrics/metrics_summary.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        "execution_device": str(device),
        "train_time_rf_sec": float(rf_train_time),
        "train_time_gru_sec": float(gru_train_time),
        "train_time_tcn_sec": float(tcn_train_time),
        "thresholds": [float(splits.thresholds[0]), float(splits.thresholds[1])],
        "baseline_rf": {
            "regression": rf_eval["regression"],
            "classification": {
                "precision": rf_eval["classification"]["precision"],
                "recall": rf_eval["classification"]["recall"],
                "macro_f1": rf_eval["classification"]["macro_f1"],
                "accuracy": rf_eval["classification"]["accuracy"],
                "confusion_matrix": rf_eval["classification"]["confusion_matrix"].tolist(),
            },
        },
        "flowcast_gru": {
            "regression": gru_eval["regression"],
            "classification": {
                "precision": gru_eval["classification"]["precision"],
                "recall": gru_eval["classification"]["recall"],
                "macro_f1": gru_eval["classification"]["macro_f1"],
                "accuracy": gru_eval["classification"]["accuracy"],
                "confusion_matrix": gru_eval["classification"]["confusion_matrix"].tolist(),
            },
        },
        "flowcast_tcn_attention": {
            "regression": tcn_eval["regression"],
            "classification": {
                "precision": tcn_eval["classification"]["precision"],
                "recall": tcn_eval["classification"]["recall"],
                "macro_f1": tcn_eval["classification"]["macro_f1"],
                "accuracy": tcn_eval["classification"]["accuracy"],
                "confusion_matrix": tcn_eval["classification"]["confusion_matrix"].tolist(),
            },
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info("Metrics summary written to %s", metrics_path)

    # 8. Generate & Save Evaluation Figures
    figures_dir = Path("artifacts/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(
        tcn_train_losses,
        tcn_val_losses,
        save_path=figures_dir / "loss_curves.png",
    )
    plot_actual_vs_predicted(
        splits.Y_test,
        tcn_test_preds,
        sensor_id=0,
        time_window=200,
        save_path=figures_dir / "actual_vs_predicted_sensor0.png",
    )
    class_labels = ["Low Volume", "Moderate Volume", "High Volume"]
    plot_confusion_matrix(
        rf_eval["classification"]["confusion_matrix"],
        class_names=class_labels,
        title="Random Forest Baseline - Confusion Matrix",
        save_path=figures_dir / "confusion_matrix_baseline.png",
    )
    plot_confusion_matrix(
        tcn_eval["classification"]["confusion_matrix"],
        class_names=class_labels,
        title="FlowCast TCN-Attention - Confusion Matrix",
        save_path=figures_dir / "confusion_matrix_tcn.png",
    )
    plot_three_model_comparison(
        rf_eval=rf_eval,
        gru_eval=gru_eval,
        tcn_eval=tcn_eval,
        save_path=figures_dir / "model_comparison.png",
    )
    logger.info("Generated evaluation plots in %s", figures_dir)

    # 9. Extract multi-modal demonstration samples
    outputs_dir = Path("artifacts/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = [10, 50, 120, 250, 400]
    demo_samples = []

    for idx in sample_indices:
        pred_vol = tcn_test_preds[idx]
        actual_vol = splits.Y_test[idx]
        co2_kg = compute_carbon_emissions(pred_vol)
        surge_idx = compute_ev_grid_surge_index(pred_vol)
        advisory = compute_green_wave_advisory(pred_vol)

        demo_samples.append({
            "test_sample_index": int(idx),
            "actual_mean_volume": float(np.mean(actual_vol)),
            "predicted_mean_volume": float(np.mean(pred_vol)),
            "predicted_co2_kg_total": float(np.sum(co2_kg)),
            "mean_ev_grid_surge_index": float(np.mean(surge_idx)),
            "recommended_speed_mean_kmh": float(np.mean(advisory["recommended_speed_kmh"])),
            "metering_active_sensors": int(np.sum(advisory["metering_active"])),
            "sensor_actual": actual_vol.tolist(),
            "sensor_predicted": pred_vol.tolist(),
            "sensor_co2_kg": co2_kg.tolist(),
            "sensor_ev_surge": surge_idx.tolist(),
        })

    with open(outputs_dir / "demo_samples.json", "w", encoding="utf-8") as f:
        json.dump(demo_samples, f, indent=2)
    logger.info("Demonstration samples with multi-modal metrics written to demo_samples.json")

    return {
        "rf_eval": rf_eval,
        "gru_eval": gru_eval,
        "tcn_eval": tcn_eval,
        "metrics_summary": metrics_data,
    }
