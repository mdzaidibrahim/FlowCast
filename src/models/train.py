"""End-to-end training and evaluation pipeline for FlowCast.

Trains Random Forest baseline and FlowCastGRU with early stopping,
logs training curves, evaluates test performance, and saves all artifacts.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.preprocess import prepare_data_splits
from ..evaluation.metrics import evaluate_model_performance
from ..evaluation.plots import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_curves,
)
from .baseline import FlowCastBaseline
from .gru_model import FlowCastGRU, get_default_device

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_gru(
    splits,
    device: torch.device,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 0.002,
    patience: int = 12,
    checkpoint_path: Path = Path("artifacts/models/gru_best.pth"),
) -> Tuple[FlowCastGRU, list[float], list[float]]:
    """Train FlowCastGRU with early stopping, LR scheduling, and model checkpointing."""
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
                "Epoch [%02d/%02d] | Train Loss: %.5f | Val Loss: %.5f %s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                status_tag,
            )

        if patience_counter >= patience:
            logger.info(
                "Early stopping triggered at epoch %d. Best Val Loss: %.5f",
                epoch,
                best_val_loss,
            )
            break

    # Load best checkpoint before returning
    best_model = FlowCastGRU.load(checkpoint_path, device=device)
    return best_model, train_losses, val_losses


def run_pipeline() -> Dict[str, Any]:
    """Execute complete data loading, model training, evaluation, and artifact generation."""
    set_seed(42)
    device = get_default_device()
    logger.info("Starting FlowCast training pipeline on device: %s", device)

    # 1. Prepare data splits
    splits = prepare_data_splits(val_size=200)
    logger.info(
        "Data splits ready. Train: %d, Val: %d, Test: %d timestamps.",
        len(splits.X_train_seq),
        len(splits.X_val_seq),
        len(splits.X_test_seq),
    )
    logger.info("Classification thresholds (derived from train): %s", splits.thresholds)

    # 2. Train baseline
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
    gru_model, train_losses, val_losses = train_gru(
        splits, device=device, epochs=60, batch_size=32, lr=0.002, checkpoint_path=gru_path
    )
    gru_train_time = time.time() - t0
    logger.info("GRU model training finished in %.2fs. Best model at %s", gru_train_time, gru_path)

    # 4. Generate Predictions on Held-Out Test Set
    rf_test_preds = rf.predict(splits.X_test_flat)

    test_seq = torch.tensor(splits.X_test_seq, dtype=torch.float32).to(device)
    test_ctx = torch.tensor(splits.X_test_ctx, dtype=torch.float32).to(device)
    gru_model.eval()
    with torch.no_grad():
        gru_test_preds = gru_model(test_seq, test_ctx).cpu().numpy()

    # 5. Evaluate Metrics
    rf_eval = evaluate_model_performance(
        splits.Y_test, rf_test_preds, splits.thresholds, model_name="Random Forest"
    )
    gru_eval = evaluate_model_performance(
        splits.Y_test, gru_test_preds, splits.thresholds, model_name="FlowCast GRU"
    )

    logger.info("=== TEST EVALUATION RESULTS ===")
    logger.info(
        "Random Forest -> MAE: %.4f, RMSE: %.4f, R2: %.4f | Macro-F1: %.4f, Acc: %.4f",
        rf_eval["regression"]["mae"],
        rf_eval["regression"]["rmse"],
        rf_eval["regression"]["r2"],
        rf_eval["classification"]["macro_f1"],
        rf_eval["classification"]["accuracy"],
    )
    logger.info(
        "FlowCast GRU  -> MAE: %.4f, RMSE: %.4f, R2: %.4f | Macro-F1: %.4f, Acc: %.4f",
        gru_eval["regression"]["mae"],
        gru_eval["regression"]["rmse"],
        gru_eval["regression"]["r2"],
        gru_eval["classification"]["macro_f1"],
        gru_eval["classification"]["accuracy"],
    )

    # 6. Save Metrics Summary JSON
    metrics_path = Path("artifacts/metrics/metrics_summary.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        "execution_device": str(device),
        "train_time_rf_sec": float(rf_train_time),
        "train_time_gru_sec": float(gru_train_time),
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
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info("Saved metrics summary to %s", metrics_path)

    # 7. Generate and Save Figures
    fig_dir = Path("artifacts/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(
        train_losses, val_losses, save_path=fig_dir / "loss_curves.png"
    )
    plot_actual_vs_predicted(
        splits.Y_test, gru_test_preds, sensor_id=0, save_path=fig_dir / "actual_vs_predicted_sensor0.png"
    )
    plot_confusion_matrix(
        gru_eval["classification"]["confusion_matrix"],
        gru_eval["classification"]["class_names"],
        title="FlowCast GRU: Traffic Band Confusion Matrix",
        save_path=fig_dir / "confusion_matrix_gru.png",
    )
    plot_confusion_matrix(
        rf_eval["classification"]["confusion_matrix"],
        rf_eval["classification"]["class_names"],
        title="Random Forest: Traffic Band Confusion Matrix",
        save_path=fig_dir / "confusion_matrix_baseline.png",
    )
    plot_model_comparison(
        rf_eval["regression"],
        gru_eval["regression"],
        rf_eval["classification"],
        gru_eval["classification"],
        save_path=fig_dir / "model_comparison.png",
    )
    logger.info("Generated all diagnostic plots in %s", fig_dir)

    # 8. Save Sample Test Demonstrations and Sensor Metadata for Streamlit Loading
    out_dir = Path("artifacts/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save sensor metadata for exact flattened feature reconstruction
    raw_data = load_raw_traffic_data()
    X0 = raw_data["X_tr_raw"][0]
    sensor_meta = {
        "num_lanes": X0[:, 39].astype(int).tolist(),
        "road_direction": np.argmax(X0[:, 40:44], axis=1).tolist(),
        "road_id": np.argmax(X0[:, 44:48], axis=1).tolist(),
    }
    with open(out_dir / "sensor_metadata.json", "w") as f:
        json.dump(sensor_meta, f, indent=2)

    sample_data = {
        "sample_indices": [0, 50, 100, 200, 300, 400],
        "thresholds": [float(splits.thresholds[0]), float(splits.thresholds[1])],
        "class_names": ["Low Volume", "Moderate Volume", "High Volume"],
        "num_sensors": 36,
        "samples": [],
    }
    for idx in sample_data["sample_indices"]:
        sample_data["samples"].append(
            {
                "test_index": idx,
                "x_seq": splits.X_test_seq[idx].tolist(),  # (10, 36)
                "x_ctx": splits.X_test_ctx[idx].tolist(),  # (29,)
                "x_flat": splits.X_test_flat[idx].tolist(),  # (1728,)
                "y_true": splits.Y_test[idx].tolist(),  # (36,)
                "gru_pred": gru_test_preds[idx].tolist(),
                "rf_pred": rf_test_preds[idx].tolist(),
            }
        )
    with open(out_dir / "demo_samples.json", "w") as f:
        json.dump(sample_data, f)
    logger.info("Saved demonstration samples and sensor metadata to %s", out_dir)

    return metrics_data


if __name__ == "__main__":
    run_pipeline()
