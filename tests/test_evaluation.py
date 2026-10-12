"""Evaluation and metrics tests for FlowCast.

Tests per-sensor regression metrics, classification thresholds behaviour,
confusion-matrix consistency, and the predict module output contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.preprocess import classify_traffic, prepare_data_splits
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_per_sensor_regression_metrics,
    compute_regression_metrics,
    evaluate_model_performance,
)
from src.models.gru_model import FlowCastGRU
from src.models.predict import FlowCastPredictor


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def splits():
    """Load pre-processed data splits once per module."""
    return prepare_data_splits(val_size=200)


@pytest.fixture(scope="module")
def predictor():
    """Load saved predictor once per module."""
    return FlowCastPredictor(
        gru_path="artifacts/models/gru_best.pth",
        rf_path="artifacts/models/baseline_rf.joblib",
        metrics_path="artifacts/metrics/metrics_summary.json",
    )


# ──────────────────────────────────────────────────────────
# Regression metrics
# ──────────────────────────────────────────────────────────

def test_regression_metrics_perfect_prediction():
    """MAE=0, RMSE=0, R²=1 for identical arrays."""
    y = np.random.rand(100, 36).astype(np.float32)
    metrics = compute_regression_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)


def test_regression_metrics_range(splits):
    """Network-wide MAE/RMSE must be non-negative; R² in plausible range."""
    y_true = splits.Y_test
    y_pred = np.clip(y_true + np.random.normal(0, 0.05, y_true.shape), 0.0, 1.0)
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert -1.0 <= metrics["r2"] <= 1.0


def test_per_sensor_regression_returns_all_sensors(splits):
    """Per-sensor metrics dict must contain an entry for each of the 36 sensors."""
    y_true = splits.Y_test
    y_pred = y_true + np.random.normal(0, 0.02, y_true.shape)
    per_sensor = compute_per_sensor_regression_metrics(y_true, y_pred)
    assert len(per_sensor) == 36
    for sid, m in per_sensor.items():
        assert 0 <= sid <= 35
        assert "mae" in m and "rmse" in m and "r2" in m
        assert m["mae"] >= 0.0
        assert m["rmse"] >= 0.0


# ──────────────────────────────────────────────────────────
# Classification metrics
# ──────────────────────────────────────────────────────────

def test_classify_traffic_boundary_conditions():
    """Exact threshold values must map to correct class boundaries."""
    thresholds = (0.3, 0.6)
    volumes = np.array([0.0, 0.3, 0.3001, 0.6, 0.6001, 1.0])
    cats = classify_traffic(volumes, thresholds)
    assert cats[0] == 0   # 0.0 → Low
    assert cats[1] == 0   # 0.3 == low_th → Low (≤ not <)
    assert cats[2] == 1   # 0.3001 → Moderate
    assert cats[3] == 1   # 0.6 == high_th → Moderate
    assert cats[4] == 2   # 0.6001 → High
    assert cats[5] == 2   # 1.0 → High


def test_classify_traffic_only_valid_classes():
    """All classification outputs must lie in {0, 1, 2}."""
    thresholds = (0.2, 0.5)
    volumes = np.random.rand(500).astype(np.float32)
    cats = classify_traffic(volumes, thresholds)
    assert set(np.unique(cats)).issubset({0, 1, 2})


def test_classification_metrics_perfect(splits):
    """Perfect regression forecasts should yield classification accuracy near 1.0."""
    y_true = splits.Y_test
    # Perfect prediction
    metrics = compute_classification_metrics(y_true, y_true, splits.thresholds)
    assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["macro_f1"] == pytest.approx(1.0, abs=1e-6)
    assert len(metrics["class_names"]) == 3


def test_confusion_matrix_sum_matches_total_elements(splits):
    """Confusion matrix row sums must equal the total number of sensor readings."""
    y_true = splits.Y_test
    y_pred = y_true + np.random.normal(0, 0.05, y_true.shape)
    metrics = compute_classification_metrics(y_true, y_pred, splits.thresholds)
    cm = metrics["confusion_matrix"]
    assert cm.shape == (3, 3)
    # Total predictions == total ground-truth labels
    assert int(cm.sum()) == y_true.size


# ──────────────────────────────────────────────────────────
# Full evaluate_model_performance contract
# ──────────────────────────────────────────────────────────

def test_evaluate_model_performance_keys(splits):
    """evaluate_model_performance must return regression and classification sub-dicts."""
    y_true = splits.Y_test
    y_pred = np.clip(y_true + 0.01, 0.0, 1.0)
    result = evaluate_model_performance(y_true, y_pred, splits.thresholds, model_name="Test")
    assert result["model_name"] == "Test"
    assert "mae" in result["regression"]
    assert "rmse" in result["regression"]
    assert "r2" in result["regression"]
    assert "macro_f1" in result["classification"]
    assert "accuracy" in result["classification"]
    assert "confusion_matrix" in result["classification"]


# ──────────────────────────────────────────────────────────
# GRU model architecture invariants
# ──────────────────────────────────────────────────────────

def test_gru_output_bounded_sigmoid():
    """GRU forward output must always be in [0, 1] due to Sigmoid activation."""
    model = FlowCastGRU(num_sensors=36, seq_len=10, hidden_dim=32, num_layers=2, context_dim=16)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            # Use extreme random inputs to stress-test the activation boundary
            x_seq = torch.randn(8, 10, 36) * 10
            x_ctx = torch.randn(8, 29) * 10
            out = model(x_seq, x_ctx)
            assert (out >= 0.0).all() and (out <= 1.0).all(), "Sigmoid bound violated"


def test_gru_checkpoint_architecture_preserved(tmp_path):
    """Saved checkpoint must exactly restore original architecture hyperparameters."""
    original = FlowCastGRU(
        num_sensors=36, seq_len=10, hidden_dim=48, num_layers=3,
        context_dim=24, dropout=0.2
    )
    save_path = tmp_path / "test_checkpoint.pth"
    original.save(save_path)

    loaded = FlowCastGRU.load(save_path)
    assert loaded.hidden_dim == 48
    assert loaded.num_layers == 3
    assert loaded.context_dim == 24
    assert abs(loaded.dropout_rate - 0.2) < 1e-6


# ──────────────────────────────────────────────────────────
# FlowCastPredictor inference
# ──────────────────────────────────────────────────────────

def test_predictor_gru_batch_consistency(predictor):
    """GRU results must be identical when the same sample is fed as a batch of 3."""
    np.random.seed(7)
    x_seq = np.random.rand(10, 36).astype(np.float32)
    x_ctx = np.random.rand(29).astype(np.float32)

    # Single-sample inference (auto-batched internally)
    single = predictor.predict_gru(x_seq, x_ctx)[0]  # (36,)

    # 3-sample batch with identical inputs
    x_seq_batch = np.stack([x_seq] * 3, axis=0)
    x_ctx_batch = np.stack([x_ctx] * 3, axis=0)
    batch = predictor.predict_gru(x_seq_batch, x_ctx_batch)  # (3, 36)

    np.testing.assert_allclose(single, batch[0], atol=1e-5)
    np.testing.assert_allclose(batch[0], batch[1], atol=1e-5)


def test_predictor_category_labels_valid(predictor):
    """All category labels returned must belong to the three valid class strings."""
    valid_labels = {"Low Volume", "Moderate Volume", "High Volume"}
    np.random.seed(0)
    x_seq = np.random.rand(10, 36).astype(np.float32)
    x_ctx = np.random.rand(29).astype(np.float32)
    result = predictor.forecast_and_classify(x_seq, x_ctx, model_type="gru", sensor_id=5)
    assert set(result["all_categories"]).issubset(valid_labels)
    assert result["sensor_category"] in valid_labels
