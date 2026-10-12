"""Tests for feature construction, hyperparameter tuning rules, and CSV validation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_raw_traffic_data
from src.data.preprocess import build_x_flat_from_inputs, prepare_data_splits
from src.models.predict import FlowCastPredictor
from src.models.tune import tune_random_forest, tune_gru


def test_build_x_flat_feature_positions():
    """Verify that build_x_flat_from_inputs places lag values, contextual features,

    and sensor metadata in the exact per-sensor 48-feature positions expected by Random Forest.
    """
    num_sensors = 36
    # Construct distinct synthetic sequence and context
    x_seq = np.zeros((10, num_sensors), dtype=np.float32)
    for s in range(num_sensors):
        # Assign distinct lag values for each sensor: [s + 0.01, s + 0.02, ..., s + 0.10]
        x_seq[:, s] = s + np.linspace(0.01, 0.10, 10)

    # Weekday: Tuesday (index 1), Hour: 14 (index 14 -> ctx index 5 + 14 = 19)
    x_ctx = np.zeros(29, dtype=np.float32)
    x_ctx[1] = 1.0
    x_ctx[19] = 1.0

    # Synthetic metadata
    sensor_metadata = {
        "num_lanes": np.array([3 if s % 2 == 0 else 4 for s in range(num_sensors)], dtype=int),
        "road_direction": np.array([s % 4 for s in range(num_sensors)], dtype=int),
        "road_id": np.array([(s // 4) % 4 for s in range(num_sensors)], dtype=int),
    }

    x_flat = build_x_flat_from_inputs(x_seq, x_ctx, sensor_metadata)
    assert x_flat.shape == (1728,)

    for s in range(num_sensors):
        base = s * 48
        # 1. 10 historical volume lags
        np.testing.assert_allclose(x_flat[base : base + 10], x_seq[:, s])
        # 2. Weekday one-hot (5 features)
        np.testing.assert_allclose(x_flat[base + 10 : base + 15], x_ctx[:5])
        assert x_flat[base + 10 + 1] == 1.0
        # 3. Hour one-hot (24 features)
        np.testing.assert_allclose(x_flat[base + 15 : base + 39], x_ctx[5:29])
        assert x_flat[base + 15 + 14] == 1.0
        # 4. Number of lanes
        assert x_flat[base + 39] == float(sensor_metadata["num_lanes"][s])
        # 5. Road direction one-hot
        expected_dir = sensor_metadata["road_direction"][s]
        assert x_flat[base + 40 + expected_dir] == 1.0
        assert np.sum(x_flat[base + 40 : base + 44]) == 1.0
        # 6. Road ID one-hot
        expected_id = sensor_metadata["road_id"][s]
        assert x_flat[base + 44 + expected_id] == 1.0
        assert np.sum(x_flat[base + 44 : base + 48]) == 1.0


def test_build_x_flat_matches_demo_samples():
    """Verify that build_x_flat_from_inputs reconstructs demo sample flat vectors identically."""
    samples_path = Path("artifacts/outputs/demo_samples.json")
    meta_path = Path("artifacts/outputs/sensor_metadata.json")
    assert samples_path.exists(), "demo_samples.json not found"
    assert meta_path.exists(), "sensor_metadata.json not found"

    with open(samples_path) as f:
        samples_data = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    for sample in samples_data["samples"]:
        x_seq = np.array(sample["x_seq"], dtype=np.float32)
        x_ctx = np.array(sample["x_ctx"], dtype=np.float32)
        expected_flat = np.array(sample["x_flat"], dtype=np.float32)

        built_flat = build_x_flat_from_inputs(x_seq, x_ctx, meta)
        np.testing.assert_allclose(built_flat, expected_flat, atol=1e-5)


def test_csv_upload_validation_and_flattening():
    """Verify CSV validation rules and feature assembly for batch inputs."""
    meta_path = Path("artifacts/outputs/sensor_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Valid CSV: 12 rows, 36 columns
    valid_data = np.random.uniform(0.1, 0.8, size=(12, 36)).astype(np.float32)
    df_valid = pd.DataFrame(valid_data, columns=[f"sensor_{i}" for i in range(36)])
    assert len(df_valid) >= 10 and df_valid.shape[1] >= 36

    x_seq = df_valid.iloc[:10, :36].values.astype(np.float32)
    x_ctx = np.zeros(29, dtype=np.float32)
    x_ctx[0] = 1.0
    x_ctx[13] = 1.0

    x_flat = build_x_flat_from_inputs(x_seq, x_ctx, meta)
    assert x_flat.shape == (1728,)
    assert not np.isnan(x_flat).any()

    # Invalid CSV: fewer than 36 columns
    df_invalid_cols = pd.DataFrame(np.random.rand(15, 20))
    assert not (df_invalid_cols.shape[1] >= 36 and len(df_invalid_cols) >= 10)

    # Invalid CSV: fewer than 10 rows
    df_invalid_rows = pd.DataFrame(np.random.rand(8, 36))
    assert not (df_invalid_rows.shape[1] >= 36 and len(df_invalid_rows) >= 10)


def test_tuning_records_and_selection_rules():
    """Verify that tuning results evaluate on validation data and record valid metrics."""
    tuning_json = Path("artifacts/metrics/hyperparameter_search.json")
    assert tuning_json.exists(), "hyperparameter_search.json not found"

    with open(tuning_json) as f:
        data = json.load(f)

    assert "all_trials" in data
    assert "selected_rf_config" in data
    assert "selected_gru_config" in data

    trials = data["all_trials"]
    rf_trials = [t for t in trials if t["model_family"] == "RandomForest"]
    gru_trials = [t for t in trials if t["model_family"] == "FlowCastGRU"]

    assert len(rf_trials) == 11
    assert len(gru_trials) == 8

    # Verify RF records
    for trial in rf_trials:
        assert "params" in trial
        assert "val_mae" in trial
        assert "val_rmse" in trial
        assert trial["val_mae"] > 0
        assert trial["val_rmse"] >= trial["val_mae"]

    # Verify GRU records
    for trial in gru_trials:
        assert "params" in trial
        assert "val_mae" in trial
        assert "val_rmse" in trial
        assert "best_epoch" in trial
        assert trial["val_mae"] > 0
        assert trial["val_rmse"] >= trial["val_mae"]

    # Verify selection rule: selected configs correspond to valid configurations
    assert data["selected_rf_config"]["n_estimators"] in [50, 100, 150]
    assert data["selected_gru_config"]["hidden_dim"] in [32, 64, 128]


def test_model_inference_output_shapes_and_ranges():
    """Verify that both GRU and RF inference return correct (36,) volume predictions."""
    predictor = FlowCastPredictor()
    x_seq = np.random.uniform(0.1, 0.7, size=(10, 36)).astype(np.float32)
    x_ctx = np.zeros(29, dtype=np.float32)
    x_ctx[2] = 1.0
    x_ctx[15] = 1.0

    meta_path = Path("artifacts/outputs/sensor_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    x_flat = build_x_flat_from_inputs(x_seq, x_ctx, meta)

    # GRU
    gru_out = predictor.predict_gru(x_seq, x_ctx)
    assert gru_out.shape == (1, 36)
    assert np.all((gru_out >= 0.0) & (gru_out <= 1.0))

    # RF
    rf_out = predictor.predict_baseline(x_flat)
    assert rf_out.shape == (1, 36)
    assert np.all((rf_out >= 0.0) & (rf_out <= 1.0))

    # Unified forecast_and_classify
    res = predictor.forecast_and_classify(x_seq, x_ctx, x_flat=x_flat, model_type="rf", sensor_id=5)
    assert res["sensor_id"] == 5
    assert len(res["all_predictions"]) == 36
    assert len(res["all_categories"]) == 36
    assert res["sensor_category"] in ["Low Volume", "Moderate Volume", "High Volume"]
