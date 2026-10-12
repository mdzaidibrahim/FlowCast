"""Preprocessing and temporal data splitting module for FlowCast.

Enforces zero-leakage temporal ordering, derives training-only thresholds,
and formats sequential and contextual tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .load_data import load_raw_traffic_data


@dataclass
class TrafficDataSplits:
    """Container holding clean, temporally split datasets for training and evaluation."""

    # Sequential representations for GRU: shape (N, 10, 36)
    X_train_seq: np.ndarray
    X_val_seq: np.ndarray
    X_test_seq: np.ndarray

    # Contextual representations for GRU context branch: shape (N, 29)
    X_train_ctx: np.ndarray
    X_val_ctx: np.ndarray
    X_test_ctx: np.ndarray

    # Flattened representations for Random Forest baseline: shape (N, 1728)
    X_train_flat: np.ndarray
    X_val_flat: np.ndarray
    X_test_flat: np.ndarray

    # Ground-truth targets: shape (N, 36)
    Y_train: np.ndarray
    Y_val: np.ndarray
    Y_test: np.ndarray

    # Spatial highway graph adjacency matrix: shape (36, 36)
    adj_matrix: np.ndarray

    # Derived non-leaking classification thresholds (low_threshold, high_threshold)
    thresholds: Tuple[float, float]

    # Sensor attributes (num_lanes, road_dir, road_id) for UI & inspection
    sensor_metadata: Dict[str, np.ndarray]


def classify_traffic(
    volumes: np.ndarray,
    thresholds: Tuple[float, float],
) -> np.ndarray:
    """Categorize continuous traffic volume into 3 discrete traffic-volume bands.

    Classes:
        0: Low Traffic Volume (volume <= low_threshold)
        1: Moderate Traffic Volume (low_threshold < volume <= high_threshold)
        2: High Traffic Volume (volume > high_threshold)

    Args:
        volumes: Continuous traffic volume values in [0.0, 1.0].
        thresholds: (low_threshold, high_threshold) fitted on training data.

    Returns:
        Integer array of shape matching `volumes` with values in {0, 1, 2}.
    """
    low_th, high_th = thresholds
    categories = np.zeros_like(volumes, dtype=np.int64)
    categories[(volumes > low_th) & (volumes <= high_th)] = 1
    categories[volumes > high_th] = 2
    return categories


def extract_features(
    X_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sequential, contextual, and flattened representations from raw arrays.

    Args:
        X_raw: Array of shape (N, 36, 48).

    Returns:
        X_seq: Shape (N, 10, 36) - 10-step sequence across 36 sensor volumes.
        X_ctx: Shape (N, 29) - 5 weekday + 24 hour-of-day contextual features.
        X_flat: Shape (N, 1728) - Flattened feature vectors for tree baselines.
    """
    N, num_sensors, num_features = X_raw.shape

    # X_raw[:, :, :10] is (N, 36, 10) -> transpose axis 1 and 2 to get (N, 10, 36)
    X_seq = np.transpose(X_raw[:, :, :10], (0, 2, 1)).copy()

    # Time context is identical across all sensors at timestamp t:
    # Cols 10:15 (5 days) and Cols 15:39 (24 hours)
    X_ctx = X_raw[:, 0, 10:39].copy()

    # Flattened representation for Random Forest
    X_flat = X_raw.reshape(N, num_sensors * num_features).copy()

    return X_seq, X_ctx, X_flat


def build_x_flat_from_inputs(
    x_seq: np.ndarray,
    x_ctx: np.ndarray,
    sensor_metadata: Dict[str, np.ndarray],
) -> np.ndarray:
    """Assemble the correct 1728-element flattened feature vector for RF inference.

    The Random Forest was trained on ``X_raw.reshape(N, 1728)`` where
    ``X_raw[n, s, :]`` is the 48-element per-sensor vector::

        [lag0..lag9,          # cols 0:10   – 10 historical volume lags
         day0..day4,          # cols 10:15  – 5 weekday one-hot flags
         hour0..hour23,       # cols 15:39  – 24 hour-of-day one-hot flags
         num_lanes,           # col  39     – integer lane count
         dir0..dir3,          # cols 40:44  – road direction one-hot
         id0..id3]            # cols 44:48  – road segment ID one-hot

    Therefore sensor *s* occupies ``x_flat[s*48 : (s+1)*48]``.
    This function assembles the vector correctly so that synthetic or
    user-provided inputs are placed at the exact positions the RF expects.

    Args:
        x_seq: Shape ``(10, 36)`` — 10 historical volume lags for all 36 sensors.
        x_ctx: Shape ``(29,)`` — 5 weekday + 24 hour-of-day one-hot features.
        sensor_metadata: Dict with keys ``'num_lanes'``, ``'road_direction'``,
            ``'road_id'`` (each length-36 integer array from the training dataset).

    Returns:
        x_flat: Shape ``(1728,)`` array matching the RF training layout.
    """
    num_sensors = 36
    num_features = 48
    x_flat = np.zeros(num_sensors * num_features, dtype=np.float32)

    num_lanes = sensor_metadata.get("num_lanes", np.zeros(num_sensors, dtype=int))
    road_dir = sensor_metadata.get("road_direction", np.zeros(num_sensors, dtype=int))
    road_id = sensor_metadata.get("road_id", np.zeros(num_sensors, dtype=int))

    for s in range(num_sensors):
        base = s * num_features
        # Cols 0:10 — historical lags (most recent = lag at t, oldest = t-9)
        x_flat[base:base + 10] = x_seq[:, s]
        # Cols 10:15 — weekday one-hot (same across all sensors)
        x_flat[base + 10:base + 15] = x_ctx[:5]
        # Cols 15:39 — hour-of-day one-hot (same across all sensors)
        x_flat[base + 15:base + 39] = x_ctx[5:29]
        # Col 39 — number of lanes (sensor-specific static feature)
        x_flat[base + 39] = float(num_lanes[s])
        # Cols 40:44 — road direction one-hot (sensor-specific)
        dir_idx = int(road_dir[s])
        if 0 <= dir_idx < 4:
            x_flat[base + 40 + dir_idx] = 1.0
        # Cols 44:48 — road segment ID one-hot (sensor-specific)
        id_idx = int(road_id[s])
        if 0 <= id_idx < 4:
            x_flat[base + 44 + id_idx] = 1.0

    return x_flat



def prepare_data_splits(
    val_size: int = 200,
    raw_dir: Path | str = "data/raw",
) -> TrafficDataSplits:
    """Prepare chronological train, validation, and test splits without leakage.

    Args:
        val_size: Number of time steps at the tail of training data for validation.
                  Default 200 steps leaves 1,061 training steps and 200 validation steps.
        raw_dir: Directory containing raw dataset files.

    Returns:
        TrafficDataSplits dataclass with all processed arrays and thresholds.
    """
    raw_data = load_raw_traffic_data(raw_dir)
    X_tr_raw = raw_data["X_tr_raw"]  # (1261, 36, 48)
    X_te_raw = raw_data["X_te_raw"]  # (840, 36, 48)
    Y_tr_raw = raw_data["Y_tr_raw"]  # (1261, 36)
    Y_te_raw = raw_data["Y_te_raw"]  # (840, 36)
    adj_matrix = raw_data["adj_matrix"]  # (36, 36)

    total_tr = X_tr_raw.shape[0]
    split_idx = total_tr - val_size
    if split_idx <= 0:
        raise ValueError(
            f"Validation size {val_size} exceeds training steps {total_tr}"
        )

    # 1. Temporal splitting of training array into Train and Val
    X_train_raw = X_tr_raw[:split_idx]
    Y_train = Y_tr_raw[:split_idx]

    X_val_raw = X_tr_raw[split_idx:]
    Y_val = Y_tr_raw[split_idx:]

    X_test_raw = X_te_raw
    Y_test = Y_te_raw

    # 2. Extract feature views
    X_train_seq, X_train_ctx, X_train_flat = extract_features(X_train_raw)
    X_val_seq, X_val_ctx, X_val_flat = extract_features(X_val_raw)
    X_test_seq, X_test_ctx, X_test_flat = extract_features(X_test_raw)

    # 3. Derive classification thresholds STRICTLY from Y_train
    low_th = float(np.percentile(Y_train, 33.3333))
    high_th = float(np.percentile(Y_train, 66.6667))
    thresholds = (low_th, high_th)

    # 4. Extract sensor static metadata from timestamp 0
    t0_sensor_data = X_tr_raw[0]  # (36, 48)
    sensor_metadata = {
        "num_lanes": t0_sensor_data[:, 39].astype(int),
        "road_direction": np.argmax(t0_sensor_data[:, 40:44], axis=1),
        "road_id": np.argmax(t0_sensor_data[:, 44:48], axis=1),
    }

    return TrafficDataSplits(
        X_train_seq=X_train_seq,
        X_val_seq=X_val_seq,
        X_test_seq=X_test_seq,
        X_train_ctx=X_train_ctx,
        X_val_ctx=X_val_ctx,
        X_test_ctx=X_test_ctx,
        X_train_flat=X_train_flat,
        X_val_flat=X_val_flat,
        X_test_flat=X_test_flat,
        Y_train=Y_train,
        Y_val=Y_val,
        Y_test=Y_test,
        adj_matrix=adj_matrix,
        thresholds=thresholds,
        sensor_metadata=sensor_metadata,
    )
