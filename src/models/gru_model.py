"""Spatio-Temporal Dual-Branch GRU Neural Network for FlowCast.

Architecture combines a recurrent sequential branch processing genuine 10-lag historical
traffic readings with an auxiliary MLP contextual branch encoding day/hour features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


def get_default_device() -> torch.device:
    """Detect available accelerator hardware with CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FlowCastGRU(nn.Module):
    """Dual-branch spatio-temporal GRU for short-horizon multi-sensor forecasting.

    Inputs:
        x_seq: Tensor of shape (B, 10, 36) containing 10 historical lags across 36 sensors.
        x_ctx: Tensor of shape (B, 29) containing 5 weekday + 24 hour-of-day contextual features.

    Outputs:
        y_pred: Tensor of shape (B, 36) forecasting traffic volume 15m ahead across all sensors.
    """

    def __init__(
        self,
        num_sensors: int = 36,
        seq_len: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        context_dim: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.dropout_rate = dropout

        # 1. Temporal Recurrent Branch (genuine 10-step sequence)
        self.gru = nn.GRU(
            input_size=num_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. Contextual MLP Branch (day of week & hour of day)
        self.context_net = nn.Sequential(
            nn.Linear(29, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )

        # 3. Spatio-Temporal Fusion & Multi-Output Head
        fused_dim = hidden_dim + context_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_sensors),
            nn.Sigmoid(),  # Traffic volumes are strictly bounded in [0.0, 1.0]
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_seq: (B, 10, 36)
            x_ctx: (B, 29)

        Returns:
            (B, 36)
        """
        # GRU forward
        _, h_n = self.gru(x_seq)
        h_temp = h_n[-1]  # Take top layer hidden state: (B, hidden_dim)

        # Context forward
        h_ctx = self.context_net(x_ctx)  # (B, context_dim)

        # Fusion
        fused = torch.cat([h_temp, h_ctx], dim=1)  # (B, hidden_dim + context_dim)
        y_pred = self.fusion_head(fused)  # (B, num_sensors)
        return y_pred

    def save(self, filepath: Path | str) -> Path:
        """Save model state dictionary along with architectural hyperparameters."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "num_sensors": self.num_sensors,
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "context_dim": self.context_dim,
                "dropout": self.dropout_rate,
            },
        }
        torch.save(checkpoint, str(filepath))
        return filepath

    @classmethod
    def load(
        cls,
        filepath: Path | str,
        device: Optional[torch.device] = None,
    ) -> FlowCastGRU:
        """Load model from saved checkpoint."""
        if device is None:
            device = get_default_device()
        checkpoint = torch.load(str(filepath), map_location=device, weights_only=True)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
