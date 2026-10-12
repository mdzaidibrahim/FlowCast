"""Comprehensive test suite for FlowCast data loading, preprocessing, models, and inference."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.load_data import load_raw_traffic_data
from src.data.preprocess import classify_traffic, prepare_data_splits
from src.models.baseline import FlowCastBaseline
from src.models.gru_model import FlowCastGRU
from src.models.predict import FlowCastPredictor


def test_data_loader_structure():
    """Verify raw data loading dimensions and metadata."""
    raw = load_raw_traffic_data("data/raw")
    assert "X_tr_raw" in raw
    assert "X_te_raw" in raw
    assert "Y_tr_raw" in raw
    assert "Y_te_raw" in raw
    assert "adj_matrix" in raw

    assert raw["X_tr_raw"].shape == (1261, 36, 48)
    assert raw["X_te_raw"].shape == (840, 36, 48)
    assert raw["Y_tr_raw"].shape == (1261, 36)
    assert raw["Y_te_raw"].shape == (840, 36)
    assert raw["adj_matrix"].shape == (36, 36)
    assert not np.isnan(raw["Y_tr_raw"]).any()
    assert not np.isnan(raw["Y_te_raw"]).any()


def test_preprocess_splits_and_shapes():
    """Verify temporal splitting preserving non-overlapping sequence sets."""
    splits = prepare_data_splits(val_size=200)

    # Check time lengths
    assert len(splits.X_train_seq) == 1061
    assert len(splits.X_val_seq) == 200
    assert len(splits.X_test_seq) == 840
    assert len(splits.X_train_seq) + len(splits.X_val_seq) == 1261

    # Check tensor shapes
    assert splits.X_train_seq.shape == (1061, 10, 36)
    assert splits.X_train_ctx.shape == (1061, 29)
    assert splits.X_train_flat.shape == (1061, 1728)
    assert splits.Y_train.shape == (1061, 36)

    assert splits.X_val_seq.shape == (200, 10, 36)
    assert splits.X_val_ctx.shape == (200, 29)

    assert splits.X_test_seq.shape == (840, 10, 36)
    assert splits.X_test_ctx.shape == (840, 29)
    assert splits.X_test_flat.shape == (840, 1728)
    assert splits.Y_test.shape == (840, 36)


def test_no_data_leakage():
    """Verify classification thresholds are derived strictly from training data."""
    splits = prepare_data_splits(val_size=200)
    low_th, high_th = splits.thresholds

    # Thresholds should match training tertiles exactly
    expected_low = float(np.percentile(splits.Y_train, 33.3333))
    expected_high = float(np.percentile(splits.Y_train, 66.6667))
    assert np.isclose(low_th, expected_low, atol=1e-5)
    assert np.isclose(high_th, expected_high, atol=1e-5)

    # Test categorization values are in {0, 1, 2}
    cats = classify_traffic(splits.Y_test, splits.thresholds)
    assert set(np.unique(cats)).issubset({0, 1, 2})


def test_gru_forward_and_output_shapes():
    """Verify PyTorch GRU model forward pass handles arbitrary batch sizes."""
    model = FlowCastGRU(num_sensors=36, seq_len=10, hidden_dim=32, num_layers=1, context_dim=16)
    model.eval()

    batch_size = 5
    x_seq = torch.rand(batch_size, 10, 36)
    x_ctx = torch.rand(batch_size, 29)

    with torch.no_grad():
        out = model(x_seq, x_ctx)

    assert out.shape == (batch_size, 36)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_saved_model_loading_and_inference():
    """Verify saved model weights can be loaded from disk and make valid forecasts."""
    predictor = FlowCastPredictor(
        gru_path="artifacts/models/gru_best.pth",
        rf_path="artifacts/models/baseline_rf.joblib",
        metrics_path="artifacts/metrics/metrics_summary.json",
    )

    demo_file = Path("artifacts/outputs/demo_samples.json")
    assert demo_file.exists(), "Demo samples file should exist"
    with open(demo_file, "r") as f:
        demo_data = json.load(f)

    sample = demo_data["samples"][0]
    x_seq = np.array(sample["x_seq"])  # (10, 36)
    x_ctx = np.array(sample["x_ctx"])  # (29,)
    x_flat = np.array(sample["x_flat"])  # (1728,)

    # Test GRU inference
    res_gru = predictor.forecast_and_classify(x_seq, x_ctx, model_type="gru", sensor_id=0)
    assert res_gru["all_predictions"].shape == (36,)
    assert 0.0 <= res_gru["sensor_prediction"] <= 1.0
    assert res_gru["sensor_category"] in ["Low Volume", "Moderate Volume", "High Volume"]

    # Test Baseline inference
    res_rf = predictor.forecast_and_classify(x_seq, x_ctx, x_flat=x_flat, model_type="rf", sensor_id=0)
    assert res_rf["all_predictions"].shape == (36,)
    assert 0.0 <= res_rf["sensor_prediction"] <= 1.0
    assert res_rf["sensor_category"] in ["Low Volume", "Moderate Volume", "High Volume"]


def test_launcher_verification():
    """Verify run.py launcher helper functions locate files properly."""
    import run

    mat_file = run.check_and_prepare_dataset()
    assert mat_file.exists() and mat_file.stat().st_size > 1_000_000

    # Test artifact verification
    run.check_and_prepare_artifacts(force_retrain=False)
    for art in run.REQUIRED_ARTIFACTS:
        assert art.exists() and art.stat().st_size > 0
