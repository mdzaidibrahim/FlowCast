"""Inference module for FlowCast.

Loads saved model artifacts and performs short-horizon traffic forecasting
and congestion category translation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..data.preprocess import classify_traffic
from .baseline import FlowCastBaseline
from .gru_model import FlowCastGRU, get_default_device


class FlowCastPredictor:
    """Production-grade inference interface loading saved model checkpoints."""

    def __init__(
        self,
        gru_path: Path | str = "artifacts/models/gru_best.pth",
        rf_path: Path | str = "artifacts/models/baseline_rf.joblib",
        metrics_path: Path | str = "artifacts/metrics/metrics_summary.json",
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_default_device()
        self.gru_path = Path(gru_path)
        self.rf_path = Path(rf_path)
        self.metrics_path = Path(metrics_path)

        # Load thresholds
        self.thresholds: Tuple[float, float] = (0.1822, 0.3895)  # Fallback
        if self.metrics_path.exists():
            with open(self.metrics_path, "r") as f:
                data = json.load(f)
                if "thresholds" in data:
                    self.thresholds = (float(data["thresholds"][0]), float(data["thresholds"][1]))

        # Load models
        self.gru_model: Optional[FlowCastGRU] = None
        if self.gru_path.exists():
            self.gru_model = FlowCastGRU.load(self.gru_path, device=self.device)

        self.rf_model: Optional[FlowCastBaseline] = None
        if self.rf_path.exists():
            self.rf_model = FlowCastBaseline.load(self.rf_path)

        self.category_names = ["Low Volume", "Moderate Volume", "High Volume"]

    def predict_gru(
        self,
        x_seq: Union[np.ndarray, torch.Tensor],
        x_ctx: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Forecast next-15m traffic volume using FlowCastGRU.

        Args:
            x_seq: Array/Tensor of shape (B, 10, 36) or (10, 36).
            x_ctx: Array/Tensor of shape (B, 29) or (29,).

        Returns:
            NumPy array of shape (B, 36) with values in [0.0, 1.0].
        """
        if self.gru_model is None:
            raise FileNotFoundError(f"GRU model artifact not found at {self.gru_path}")

        if isinstance(x_seq, np.ndarray):
            if x_seq.ndim == 2:
                x_seq = np.expand_dims(x_seq, axis=0)
            t_seq = torch.tensor(x_seq, dtype=torch.float32).to(self.device)
        else:
            t_seq = x_seq.to(self.device)
            if t_seq.ndim == 2:
                t_seq = t_seq.unsqueeze(0)

        if isinstance(x_ctx, np.ndarray):
            if x_ctx.ndim == 1:
                x_ctx = np.expand_dims(x_ctx, axis=0)
            t_ctx = torch.tensor(x_ctx, dtype=torch.float32).to(self.device)
        else:
            t_ctx = x_ctx.to(self.device)
            if t_ctx.ndim == 1:
                t_ctx = t_ctx.unsqueeze(0)

        self.gru_model.eval()
        with torch.no_grad():
            preds = self.gru_model(t_seq, t_ctx).cpu().numpy()

        return np.clip(preds, 0.0, 1.0)

    def predict_baseline(self, x_flat: np.ndarray) -> np.ndarray:
        """Forecast next-15m traffic volume using Random Forest baseline.

        Args:
            x_flat: Array of shape (B, 1728) or (1728,).

        Returns:
            NumPy array of shape (B, 36) in [0.0, 1.0].
        """
        if self.rf_model is None:
            raise FileNotFoundError(f"Baseline model artifact not found at {self.rf_path}")
        return self.rf_model.predict(x_flat)

    def forecast_and_classify(
        self,
        x_seq: np.ndarray,
        x_ctx: np.ndarray,
        x_flat: Optional[np.ndarray] = None,
        model_type: str = "gru",
        sensor_id: int = 0,
    ) -> Dict[str, Any]:
        """Perform unified prediction, category mapping, and focal sensor extraction."""
        if model_type.lower() == "gru":
            pred_vols = self.predict_gru(x_seq, x_ctx)[0]  # shape (36,)
        elif model_type.lower() in ("rf", "baseline", "random_forest"):
            if x_flat is None:
                raise ValueError("x_flat must be provided for baseline prediction.")
            pred_vols = self.predict_baseline(x_flat)[0]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        cat_ids = classify_traffic(pred_vols, self.thresholds)
        cat_labels = [self.category_names[cid] for cid in cat_ids]

        return {
            "all_predictions": pred_vols,
            "all_categories": cat_labels,
            "category_ids": cat_ids,
            "sensor_id": sensor_id,
            "sensor_prediction": float(pred_vols[sensor_id]),
            "sensor_category": cat_labels[sensor_id],
            "thresholds": self.thresholds,
            "model_type": model_type,
            "device": str(self.device),
        }
