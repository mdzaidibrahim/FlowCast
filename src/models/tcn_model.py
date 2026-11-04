"""Temporal Convolutional Network (TCN) with Multi-Head Self-Attention.

Replaces standard recurrent networks (RNN/GRU) with dilated causal 1D convolutions
and temporal multi-head self-attention for superior long-range dependency modeling,
faster parallel training, and robust multi-sensor spatio-temporal forecasting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_default_device() -> torch.device:
    """Detect available accelerator hardware with CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ChausalConv1d(nn.Module):
    """Causal 1D Convolution with left-padding to prevent future leakage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, T)
        out = self.conv(x)
        # Trim off excess padding from the right to preserve sequence length T
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual Dilated Causal Convolutional Block with Weight Norm & Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return self.relu(out + residual)


class FlowCastTCNAttention(nn.Module):
    """Deep Hybrid TCN-Attention Spatio-Temporal Forecasting Model.

    Combines:
    1. Multi-layer Dilated Causal TCN to capture multi-scale temporal dynamics.
    2. Multi-Head Self-Attention across temporal steps to weight critical rush-hour transitions.
    3. Contextual MLP branch encoding calendar/hourly dynamics.
    4. Spatio-temporal fusion projection head with Sigmoid activation.
    """

    def __init__(
        self,
        num_sensors: int = 36,
        seq_len: int = 10,
        tcn_channels: Tuple[int, ...] = (64, 64),
        kernel_size: int = 3,
        num_heads: int = 4,
        context_dim: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.dropout_rate = dropout

        # 1. Temporal Convolutional Network Backbone
        layers = []
        in_c = num_sensors
        for i, out_c in enumerate(tcn_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_c = out_c
        self.tcn = nn.Sequential(*layers)

        # 2. Multi-Head Temporal Self-Attention
        tcn_out_dim = tcn_channels[-1]
        self.attention = nn.MultiheadAttention(
            embed_dim=tcn_out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(tcn_out_dim)

        # 3. Contextual MLP Branch (5 weekdays + 24 hours = 29 dims)
        self.context_net = nn.Sequential(
            nn.Linear(29, context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
        )

        # 4. Spatio-Temporal Fusion Head
        fused_dim = tcn_out_dim + context_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_sensors),
            nn.Sigmoid(),  # Normalized traffic volumes bounded in [0.0, 1.0]
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_seq: Tensor of shape (B, 10, 36) - batch, seq_len, num_sensors
            x_ctx: Tensor of shape (B, 29) - calendar indicators

        Returns:
            Tensor of shape (B, 36) containing forecasts for all 36 sensors.
        """
        # TCN expects (B, C, T) where C is num_sensors, T is seq_len
        x_conv = x_seq.transpose(1, 2)
        tcn_out = self.tcn(x_conv)  # (B, tcn_out_dim, T)

        # Transpose back to (B, T, C) for MultiheadAttention
        tcn_seq = tcn_out.transpose(1, 2)  # (B, 10, 64)
        attn_out, _ = self.attention(tcn_seq, tcn_seq, tcn_seq)  # (B, 10, 64)
        norm_seq = self.layer_norm(tcn_seq + attn_out)

        # Global temporal pooling (pooling the attended temporal features)
        pooled_temporal = torch.mean(norm_seq, dim=1)  # (B, 64)

        # Context branch
        ctx_features = self.context_net(x_ctx)  # (B, 32)

        # Fusion
        fused = torch.cat([pooled_temporal, ctx_features], dim=-1)  # (B, 96)
        y_pred = self.fusion_head(fused)  # (B, 36)
        return y_pred


def save_tcn_checkpoint(
    model: FlowCastTCNAttention,
    path: Path | str,
    hyperparameters: Optional[dict] = None,
) -> None:
    """Serialize model weights and architecture configuration."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "hyperparameters": hyperparameters or {},
        "model_class": "FlowCastTCNAttention",
    }
    torch.save(payload, target_path)


def load_tcn_checkpoint(
    path: Path | str,
    device: Optional[torch.device] = None,
) -> FlowCastTCNAttention:
    """Instantiate and load weights for FlowCastTCNAttention."""
    device = device or get_default_device()
    target_path = Path(path)
    checkpoint = torch.load(target_path, map_location=device)
    params = checkpoint.get("hyperparameters", {})

    model = FlowCastTCNAttention(
        num_sensors=params.get("num_sensors", 36),
        seq_len=params.get("seq_len", 10),
        tcn_channels=tuple(params.get("tcn_channels", (64, 64))),
        kernel_size=params.get("kernel_size", 3),
        num_heads=params.get("num_heads", 4),
        context_dim=params.get("context_dim", 32),
        dropout=params.get("dropout", 0.15),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
