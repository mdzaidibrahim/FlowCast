"""Inference module for FlowCast Multi-Modal Forecasting.

Loads saved checkpoints (TCN-Attention, GRU, Random Forest) and computes:
- Continuous short-horizon volume predictions across 36 sensor stations
- Volume band categorization (Low, Moderate, High)
- Multi-modal environmental indicators (CO2 emissions in kg)
- Smart grid and EV fast-charging surge pressure index
- Prescriptive green-wave speed harmonization advisories
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..data.multimodal import (
    compute_carbon_emissions,
    compute_ev_grid_surge_index,
    compute_green_wave_advisory,
)
from ..data.preprocess import classify_traffic
from .baseline import FlowCastBaseline
from .gru_model import FlowCastGRU, get_default_device
from .tcn_model import FlowCastTCNAttention, load_tcn_checkpoint


class FlowCastPredictor:
    """Production inference interface supporting multi-model and multi-modal evaluations."""

    def __init__(
        self,
        tcn_path: Path | str = "artifacts/models/tcn_attention_best.pth",
        gru_path: Path | str = "artifacts/models/gru_best.pth",
        rf_path: Path | str = "artifacts/models/baseline_rf.joblib",
        metrics_path: Path | str = "artifacts/metrics/metrics_summary.json",
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_default_device()
        self.tcn_path = Path(tcn_path)
        self.gru_path = Path(gru_path)
        self.rf_path = Path(rf_path)
        self.metrics_path = Path(metrics_path)

        # Load thresholds
        self.thresholds: Tuple[float, float] = (0.1822, 0.3895)  # Safe fallback
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "thresholds" in data:
                        self.thresholds = (float(data["thresholds"][0]), float(data["thresholds"][1]))
            except Exception:
                pass

        # Load models safely
        self.tcn_model: Optional[FlowCastTCNAttention] = None
        if self.tcn_path.exists():
            try:
                self.tcn_model = load_tcn_checkpoint(self.tcn_path, device=self.device)
            except Exception:
                pass

        self.gru_model: Optional[FlowCastGRU] = None
        if self.gru_path.exists():
            try:
                self.gru_model = FlowCastGRU.load(self.gru_path, device=self.device)
            except Exception:
                pass

        self.rf_model: Optional[FlowCastBaseline] = None
        if self.rf_path.exists():
            try:
                self.rf_model = FlowCastBaseline.load(self.rf_path)
            except Exception:
                pass

        self.category_names = ["Low Volume", "Moderate Volume", "High Volume"]

    def predict_tcn(
        self,
        x_seq: Union[np.ndarray, torch.Tensor],
        x_ctx: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Forecast 15m ahead volumes using FlowCastTCNAttention."""
        if self.tcn_model is None:
            raise FileNotFoundError(f"TCN-Attention model artifact not found at {self.tcn_path}")

        t_seq, t_ctx = self._prepare_tensors(x_seq, x_ctx)
        self.tcn_model.eval()
        with torch.no_grad():
            preds = self.tcn_model(t_seq, t_ctx).cpu().numpy()
        return np.clip(preds, 0.0, 1.0)

    def predict_gru(
        self,
        x_seq: Union[np.ndarray, torch.Tensor],
        x_ctx: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Forecast 15m ahead volumes using FlowCastGRU."""
        if self.gru_model is None:
            raise FileNotFoundError(f"GRU model artifact not found at {self.gru_path}")

        t_seq, t_ctx = self._prepare_tensors(x_seq, x_ctx)
        self.gru_model.eval()
        with torch.no_grad():
            preds = self.gru_model(t_seq, t_ctx).cpu().numpy()
        return np.clip(preds, 0.0, 1.0)

    def predict_baseline(self, x_flat: np.ndarray) -> np.ndarray:
        """Forecast next-15m traffic volume using Random Forest baseline."""
        if self.rf_model is None:
            raise FileNotFoundError(f"Baseline model artifact not found at {self.rf_path}")
        if x_flat.ndim == 1:
            x_flat = np.expand_dims(x_flat, axis=0)
        return self.rf_model.predict(x_flat)

    def _prepare_tensors(
        self,
        x_seq: Union[np.ndarray, torch.Tensor],
        x_ctx: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return t_seq, t_ctx

    def forecast_and_classify(
        self,
        x_seq: np.ndarray,
        x_ctx: np.ndarray,
        x_flat: Optional[np.ndarray] = None,
        model_type: str = "tcn",
        sensor_id: int = 0,
        ctx_hour: int = 17,
    ) -> Dict[str, Any]:
        """Perform unified multi-modal prediction and downstream urban impact calculation."""
        model_type_norm = model_type.lower()
        if "tcn" in model_type_norm:
            pred_vols = self.predict_tcn(x_seq, x_ctx)[0]  # shape (36,)
        elif "gru" in model_type_norm:
            pred_vols = self.predict_gru(x_seq, x_ctx)[0]
        elif model_type_norm in ("rf", "baseline", "random_forest"):
            if x_flat is None:
                raise ValueError("x_flat must be provided for baseline prediction.")
            pred_vols = self.predict_baseline(x_flat)[0]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        cat_ids = classify_traffic(pred_vols, self.thresholds)
        cat_labels = [self.category_names[cid] for cid in cat_ids]

        # Multi-modal calculations
        co2_emissions_kg = compute_carbon_emissions(pred_vols)
        ev_surge_index = compute_ev_grid_surge_index(pred_vols, ctx_hour=ctx_hour)
        green_wave = compute_green_wave_advisory(pred_vols)

        return {
            "all_predictions": pred_vols,
            "all_categories": cat_labels,
            "category_ids": cat_ids,
            "sensor_id": sensor_id,
            "sensor_prediction": float(pred_vols[sensor_id]),
            "sensor_category": cat_labels[sensor_id],
            "sensor_co2_kg": float(co2_emissions_kg[sensor_id]),
            "total_co2_kg": float(np.sum(co2_emissions_kg)),
            "sensor_ev_surge_index": float(ev_surge_index[sensor_id]),
            "mean_ev_surge_index": float(np.mean(ev_surge_index)),
            "recommended_speed_kmh": float(green_wave["recommended_speed_kmh"][sensor_id]),
            "all_recommended_speeds": green_wave["recommended_speed_kmh"],
            "metering_active": bool(green_wave["metering_active"][sensor_id]),
            "thresholds": self.thresholds,
            "model_type": model_type,
            "device": str(self.device),
        }
