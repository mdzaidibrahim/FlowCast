"""Hyperparameter tuning module for FlowCast.

Implements rigorous, leak-free validation-based search across Random Forest
and FlowCast GRU candidate configurations.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
logger = logging.getLogger("FlowCastTune")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Random Forest Search
# -------------------------------------------------------------------------
RF_SEARCH_GRID = [
    {"n_estimators": 50, "max_depth": 10, "max_features": "sqrt"},
    {"n_estimators": 50, "max_depth": 14, "max_features": "sqrt"},
    {"n_estimators": 50, "max_depth": 18, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 10, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 14, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 18, "max_features": "sqrt"},
    {"n_estimators": 150, "max_depth": 14, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 14, "max_features": 0.5},
    {"n_estimators": 100, "max_depth": 18, "max_features": 0.5},
    {"n_estimators": 100, "max_depth": 14, "max_features": 1.0},
    {"n_estimators": 150, "max_depth": 18, "max_features": 0.5},
]


def tune_random_forest(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Search RF candidates on training set and rank strictly by Validation MAE."""
    logger.info("Starting Random Forest hyperparameter search (%d candidates)...", len(RF_SEARCH_GRID))
    results = []
    best_config = None
    best_val_mae = float("inf")

    for i, cfg in enumerate(RF_SEARCH_GRID, start=1):
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            max_features=cfg["max_features"],
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, Y_train)
        fit_time = time.time() - t0

        val_preds = np.clip(rf.predict(X_val), 0.0, 1.0)
        val_mae = float(mean_absolute_error(Y_val, val_preds))
        val_rmse = float(np.sqrt(mean_squared_error(Y_val, val_preds)))
        val_r2 = float(r2_score(Y_val.flatten(), val_preds.flatten()))

        record = {
            "model_family": "RandomForest",
            "trial_id": f"RF_{i:02d}",
            "params": cfg,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "fit_time_sec": round(fit_time, 2),
        }
        results.append(record)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_config = cfg
            tag = "★ BEST"
        else:
            tag = ""

        logger.info(
            "[%02d/%02d] RF params: %s -> Val MAE: %.4f | Val RMSE: %.4f %s",
            i,
            len(RF_SEARCH_GRID),
            cfg,
            val_mae,
            val_rmse,
            tag,
        )

    logger.info("Selected Best RF Configuration: %s with Val MAE: %.4f", best_config, best_val_mae)
    return best_config, results


# -------------------------------------------------------------------------
# GRU Search
# -------------------------------------------------------------------------
GRU_SEARCH_GRID = [
    {
        "trial_name": "GRU_Lightweight_1L",
        "hidden_dim": 32,
        "num_layers": 1,
        "context_dim": 16,
        "dropout": 0.10,
        "lr": 0.003,
        "batch_size": 32,
    },
    {
        "trial_name": "GRU_Standard_1L",
        "hidden_dim": 64,
        "num_layers": 1,
        "context_dim": 32,
        "dropout": 0.15,
        "lr": 0.002,
        "batch_size": 32,
    },
    {
        "trial_name": "GRU_Deep_2L_Baseline",
        "hidden_dim": 64,
        "num_layers": 2,
        "context_dim": 32,
        "dropout": 0.15,
        "lr": 0.002,
        "batch_size": 32,
    },
    {
        "trial_name": "GRU_HighRegularization_2L",
        "hidden_dim": 64,
        "num_layers": 2,
        "context_dim": 32,
        "dropout": 0.25,
        "lr": 0.001,
        "batch_size": 32,
    },
    {
        "trial_name": "GRU_HighCapacity_2L",
        "hidden_dim": 128,
        "num_layers": 2,
        "context_dim": 32,
        "dropout": 0.20,
        "lr": 0.002,
        "batch_size": 32,
    },
    {
        "trial_name": "GRU_SmallBatch_2L",
        "hidden_dim": 64,
        "num_layers": 2,
        "context_dim": 32,
        "dropout": 0.15,
        "lr": 0.002,
        "batch_size": 16,
    },
    {
        "trial_name": "GRU_Compact_2L",
        "hidden_dim": 32,
        "num_layers": 2,
        "context_dim": 16,
        "dropout": 0.10,
        "lr": 0.002,
        "batch_size": 32,
    },
    {
        "trial_name": "GRU_Wide_1L_LowLR",
        "hidden_dim": 128,
        "num_layers": 1,
        "context_dim": 32,
        "dropout": 0.15,
        "lr": 0.001,
        "batch_size": 32,
    },
]


def train_single_gru_candidate(
    cfg: Dict[str, Any],
    splits,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 10,
) -> Tuple[FlowCastGRU, float, float, float, int, float, list, list]:
    """Train candidate GRU on training set; evaluate and checkpoint against validation set."""
    set_seed(42)
    model = FlowCastGRU(
        num_sensors=36,
        seq_len=10,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        context_dim=cfg["context_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    train_ds = TensorDataset(
        torch.tensor(splits.X_train_seq, dtype=torch.float32),
        torch.tensor(splits.X_train_ctx, dtype=torch.float32),
        torch.tensor(splits.Y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    val_seq = torch.tensor(splits.X_val_seq, dtype=torch.float32).to(device)
    val_ctx = torch.tensor(splits.X_val_ctx, dtype=torch.float32).to(device)
    val_y = torch.tensor(splits.Y_val, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_cnt = 0
    train_losses, val_losses = [], []

    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for b_seq, b_ctx, b_y in loader:
            b_seq, b_ctx, b_y = b_seq.to(device), b_ctx.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_seq, b_ctx)
            loss = criterion(preds, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(b_y)

        ep_tr_loss = total_loss / len(train_ds)
        train_losses.append(ep_tr_loss)

        model.eval()
        with torch.no_grad():
            v_preds = model(val_seq, val_ctx)
            v_loss = criterion(v_preds, val_y).item()
        val_losses.append(v_loss)

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    elapsed = time.time() - t0

    # Load best validation weights
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    model.eval()
    with torch.no_grad():
        final_val_preds = model(val_seq, val_ctx).cpu().numpy()

    val_mae = float(mean_absolute_error(splits.Y_val, final_val_preds))
    val_rmse = float(np.sqrt(mean_squared_error(splits.Y_val, final_val_preds)))
    val_r2 = float(r2_score(splits.Y_val.flatten(), final_val_preds.flatten()))

    return model, val_mae, val_rmse, val_r2, best_epoch, elapsed, train_losses, val_losses


def tune_gru(splits, device: torch.device) -> Tuple[Dict[str, Any], FlowCastGRU, List[Dict[str, Any]], list, list]:
    """Search GRU candidate configurations on training set, selecting strictly by Validation MAE."""
    logger.info("Starting FlowCast GRU hyperparameter search (%d candidates) on %s...", len(GRU_SEARCH_GRID), device)
    results = []
    best_config = None
    best_model = None
    best_val_mae = float("inf")
    best_tr_losses, best_v_losses = [], []

    for i, cfg in enumerate(GRU_SEARCH_GRID, start=1):
        cand_model, val_mae, val_rmse, val_r2, best_ep, elapsed, tr_l, v_l = train_single_gru_candidate(
            cfg, splits, device
        )

        record = {
            "model_family": "FlowCastGRU",
            "trial_id": cfg["trial_name"],
            "params": cfg,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "best_epoch": best_ep,
            "fit_time_sec": round(elapsed, 2),
        }
        results.append(record)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_config = cfg
            best_model = cand_model
            best_tr_losses = tr_l
            best_v_losses = v_l
            tag = "★ BEST"
        else:
            tag = ""

        logger.info(
            "[%02d/%02d] %s -> Val MAE: %.4f | Val RMSE: %.4f (Epoch %02d in %.1fs) %s",
            i,
            len(GRU_SEARCH_GRID),
            cfg["trial_name"],
            val_mae,
            val_rmse,
            best_ep,
            elapsed,
            tag,
        )

    logger.info("Selected Best GRU Configuration: %s with Val MAE: %.4f", best_config["trial_name"], best_val_mae)
    return best_config, best_model, results, best_tr_losses, best_v_losses


def plot_tuning_summary(
    rf_trials: List[Dict[str, Any]],
    gru_trials: List[Dict[str, Any]],
    save_path: Path,
) -> None:
    """Generate a clean comparative bar chart of validation MAE across tested configurations."""
    fig, (ax_rf, ax_gru) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # RF plot
    rf_names = [f"RF-{i+1}" for i in range(len(rf_trials))]
    rf_maes = [t["val_mae"] for t in rf_trials]
    min_rf = min(rf_maes)
    colors_rf = ["#2ca02c" if m == min_rf else "#aec7e8" for m in rf_maes]
    ax_rf.bar(rf_names, rf_maes, color=colors_rf)
    ax_rf.set_title("Random Forest: Validation MAE across Trials", fontsize=11, fontweight="bold")
    ax_rf.set_ylabel("Validation MAE (Lower is Better)")
    ax_rf.set_xticks(range(len(rf_names)))
    ax_rf.set_xticklabels(rf_names, rotation=45, ha="right", fontsize=9)
    ax_rf.axhline(min_rf, color="#2ca02c", linestyle="--", alpha=0.7, label=f"Best: {min_rf:.4f}")
    ax_rf.legend()

    # GRU plot
    gru_names = [t["trial_id"].replace("GRU_", "") for t in gru_trials]
    gru_maes = [t["val_mae"] for t in gru_trials]
    min_gru = min(gru_maes)
    colors_gru = ["#d62728" if m == min_gru else "#ffbb78" for m in gru_maes]
    ax_gru.bar(gru_names, gru_maes, color=colors_gru)
    ax_gru.set_title("FlowCast GRU: Validation MAE across Architectures", fontsize=11, fontweight="bold")
    ax_gru.set_ylabel("Validation MAE (Lower is Better)")
    ax_gru.set_xticks(range(len(gru_names)))
    ax_gru.set_xticklabels(gru_names, rotation=45, ha="right", fontsize=8)
    ax_gru.axhline(min_gru, color="#d62728", linestyle="--", alpha=0.7, label=f"Best: {min_gru:.4f}")
    ax_gru.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


def run_tuning_pipeline() -> Dict[str, Any]:
    """Execute complete leak-free hyperparameter search, select winners, and evaluate on test set."""
    set_seed(42)
    device = get_default_device()
    splits = prepare_data_splits(val_size=200)

    # 1. Tune Random Forest
    best_rf_cfg, rf_trials = tune_random_forest(
        splits.X_train_flat, splits.Y_train, splits.X_val_flat, splits.Y_val
    )

    # 2. Tune GRU
    best_gru_cfg, best_gru_model, gru_trials, tr_losses, v_losses = tune_gru(splits, device)

    # 3. Fit optimal RF on full training data (Train split)
    logger.info("Retraining selected best Random Forest on training data...")
    t0 = time.time()
    final_rf = FlowCastBaseline(
        n_estimators=best_rf_cfg["n_estimators"],
        max_depth=best_rf_cfg["max_depth"],
        random_state=42,
    )
    final_rf.model.set_params(max_features=best_rf_cfg["max_features"])
    final_rf.fit(splits.X_train_flat, splits.Y_train)
    rf_train_time = time.time() - t0

    # Save best models
    rf_save_path = Path("artifacts/models/baseline_rf.joblib")
    gru_save_path = Path("artifacts/models/gru_best.pth")
    final_rf.save(rf_save_path)
    best_gru_model.save(gru_save_path)

    # 4. NOW AND ONLY NOW: Evaluate winning configurations on HELD-OUT TEST SET
    logger.info("Evaluating selected configurations on held-out test horizon (N=840)...")
    rf_test_preds = final_rf.predict(splits.X_test_flat)

    test_seq = torch.tensor(splits.X_test_seq, dtype=torch.float32).to(device)
    test_ctx = torch.tensor(splits.X_test_ctx, dtype=torch.float32).to(device)
    best_gru_model.eval()
    with torch.no_grad():
        gru_test_preds = best_gru_model(test_seq, test_ctx).cpu().numpy()

    rf_eval = evaluate_model_performance(
        splits.Y_test, rf_test_preds, splits.thresholds, model_name="Tuned Random Forest"
    )
    gru_eval = evaluate_model_performance(
        splits.Y_test, gru_test_preds, splits.thresholds, model_name=f"Tuned GRU ({best_gru_cfg['trial_name']})"
    )

    # 5. Export Tuning Results Table (CSV and JSON)
    all_trials = rf_trials + gru_trials
    out_dir_metrics = Path("artifacts/metrics")
    out_dir_outputs = Path("outputs/metrics")
    out_dir_metrics.mkdir(parents=True, exist_ok=True)
    out_dir_outputs.mkdir(parents=True, exist_ok=True)

    csv_paths = [
        out_dir_metrics / "hyperparameter_search.csv",
        out_dir_outputs / "hyperparameter_search.csv",
    ]
    for cp in csv_paths:
        with open(cp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model Family", "Trial ID", "Parameters", "Val MAE", "Val RMSE", "Val R2", "Fit Time (s)"])
            for t in all_trials:
                writer.writerow([
                    t["model_family"],
                    t["trial_id"],
                    json.dumps(t["params"]),
                    f"{t['val_mae']:.5f}",
                    f"{t['val_rmse']:.5f}",
                    f"{t['val_r2']:.5f}",
                    t["fit_time_sec"],
                ])

    json_paths = [
        out_dir_metrics / "hyperparameter_search.json",
        out_dir_outputs / "hyperparameter_search.json",
    ]
    for jp in json_paths:
        with open(jp, "w") as f:
            json.dump({
                "selected_rf_config": best_rf_cfg,
                "selected_gru_config": best_gru_cfg,
                "all_trials": all_trials,
            }, f, indent=2)

    # 6. Save Tuning Diagnostic Figure
    fig_dir = Path("artifacts/figures")
    plot_tuning_summary(rf_trials, gru_trials, fig_dir / "tuning_comparison.png")

    # 7. Update authoritative metrics_summary.json
    metrics_summary_path = out_dir_metrics / "metrics_summary.json"
    metrics_summary_data = {
        "execution_device": str(device),
        "selected_rf_params": best_rf_cfg,
        "selected_gru_params": best_gru_cfg,
        "train_time_rf_sec": float(rf_train_time),
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
        "train_losses": [float(x) for x in tr_losses],
        "val_losses": [float(x) for x in v_losses],
    }
    with open(metrics_summary_path, "w") as f:
        json.dump(metrics_summary_data, f, indent=2)

    # 8. Regenerate Evaluation Plots with Tuned Models
    plot_training_curves(tr_losses, v_losses, save_path=fig_dir / "loss_curves.png")
    plot_actual_vs_predicted(
        splits.Y_test, gru_test_preds, sensor_id=0, save_path=fig_dir / "actual_vs_predicted_sensor0.png"
    )
    plot_confusion_matrix(
        gru_eval["classification"]["confusion_matrix"],
        gru_eval["classification"]["class_names"],
        title=f"FlowCast GRU ({best_gru_cfg['trial_name']}): Confusion Matrix",
        save_path=fig_dir / "confusion_matrix_gru.png",
    )
    plot_confusion_matrix(
        rf_eval["classification"]["confusion_matrix"],
        rf_eval["classification"]["class_names"],
        title="Tuned Random Forest: Confusion Matrix",
        save_path=fig_dir / "confusion_matrix_baseline.png",
    )
    plot_model_comparison(
        rf_eval["regression"],
        gru_eval["regression"],
        rf_eval["classification"],
        gru_eval["classification"],
        save_path=fig_dir / "model_comparison.png",
    )

    # 9. Update demo samples
    out_dir = Path("artifacts/outputs")
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
                "x_seq": splits.X_test_seq[idx].tolist(),
                "x_ctx": splits.X_test_ctx[idx].tolist(),
                "x_flat": splits.X_test_flat[idx].tolist(),
                "y_true": splits.Y_test[idx].tolist(),
                "gru_pred": gru_test_preds[idx].tolist(),
                "rf_pred": rf_test_preds[idx].tolist(),
            }
        )
    with open(out_dir / "demo_samples.json", "w") as f:
        json.dump(sample_data, f)

    logger.info("Hyperparameter tuning and post-selection evaluation complete!")
    return metrics_summary_data


if __name__ == "__main__":
    run_tuning_pipeline()
