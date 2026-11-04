"""Tests for FlowCast Multi-Modal environmental extensions and TCN-Attention architecture."""

import numpy as np
import pytest
import torch

from src.data.multimodal import (
    compute_carbon_emissions,
    compute_ev_grid_surge_index,
    compute_green_wave_advisory,
)
from src.models.predict import FlowCastPredictor
from src.models.tcn_model import FlowCastTCNAttention


def test_carbon_emissions_monotonicity_and_bounds():
    """Verify CO2 emissions increase non-linearly with traffic volume and idling penalties."""
    vols_low = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vols_high = np.array([0.7, 0.85, 0.95], dtype=np.float32)

    co2_low = compute_carbon_emissions(vols_low)
    co2_high = compute_carbon_emissions(vols_high)

    assert np.all(co2_low > 0.0)
    assert np.all(co2_high > co2_low)
    # Saturation non-linearity: high volume emission per car is substantially higher due to idling
    ratio = np.mean(co2_high) / np.mean(co2_low)
    assert ratio > 3.0, f"Expected non-linear congestion emission multiplier > 3.0, got {ratio}"


def test_ev_grid_surge_index_peak_hours():
    """Verify EV fast charging surge index spikes appropriately during peak evening hours."""
    vols = np.array([0.6, 0.6, 0.6], dtype=np.float32)
    surge_offpeak = compute_ev_grid_surge_index(vols, ctx_hour=13)  # 1 PM
    surge_peak = compute_ev_grid_surge_index(vols, ctx_hour=18)     # 6 PM

    assert np.all(surge_peak > surge_offpeak)
    assert np.all(surge_peak <= 100.0)
    assert np.all(surge_offpeak >= 0.0)


def test_green_wave_speed_advisories():
    """Verify speed harmonization drops dynamically during congestion to suppress shockwaves."""
    vols = np.array([0.15, 0.50, 0.85], dtype=np.float32)
    advisory = compute_green_wave_advisory(vols)

    speeds = advisory["recommended_speed_kmh"]
    metering = advisory["metering_active"]

    assert speeds[0] == 100.0  # Free flow
    assert 60.0 <= speeds[1] <= 80.0  # Moderate advisory
    assert speeds[2] <= 50.0  # Severe congestion metering
    assert not metering[0]
    assert metering[2]


def test_tcn_attention_forward_pass_and_invariance():
    """Verify FlowCastTCNAttention produces correct (B, 36) outputs strictly in [0.0, 1.0]."""
    model = FlowCastTCNAttention(
        num_sensors=36,
        seq_len=10,
        tcn_channels=(32, 32),
        num_heads=2,
    )
    x_seq = torch.randn(4, 10, 36)
    x_ctx = torch.randn(4, 29)
    out = model(x_seq, x_ctx)

    assert out.shape == (4, 36)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_predictor_multimodal_pipeline():
    """Verify FlowCastPredictor integrates TCN inference and multi-modal outputs."""
    predictor = FlowCastPredictor()
    x_seq = np.random.uniform(0.1, 0.6, size=(10, 36)).astype(np.float32)
    x_ctx = np.zeros(29, dtype=np.float32)
    x_ctx[0] = 1.0
    x_ctx[5 + 17] = 1.0  # 17:00

    res = predictor.forecast_and_classify(x_seq, x_ctx, model_type="tcn", sensor_id=5, ctx_hour=17)

    assert "all_predictions" in res
    assert len(res["all_predictions"]) == 36
    assert "sensor_co2_kg" in res
    assert "sensor_ev_surge_index" in res
    assert "recommended_speed_kmh" in res
    assert 0.0 <= res["sensor_prediction"] <= 1.0

