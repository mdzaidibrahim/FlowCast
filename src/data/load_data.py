"""Data loading module for UCI Traffic Flow Forecasting dataset (ID: 734).

Retrieves and parses the official MATLAB archive containing spatio-temporal traffic observations.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.io as sio

UCI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/734/traffic+flow+forecasting-1.zip"
)


def download_dataset_if_needed(raw_dir: Path | str = "data/raw") -> Path:
    """Download and extract the UCI Traffic Flow Forecasting dataset if not present.

    Args:
        raw_dir: Path to directory where raw files should reside.

    Returns:
        Path to the extracted traffic_dataset.mat file.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    mat_file = raw_dir / "traffic_dataset.mat"

    if mat_file.exists() and mat_file.stat().st_size > 1_000_000:
        return mat_file

    zip_file = raw_dir / "traffic_flow_forecasting.zip"
    if not zip_file.exists() or zip_file.stat().st_size < 1_000_000:
        print(f"Downloading dataset from {UCI_DOWNLOAD_URL}...")
        download_success = False

        # Method 1: requests (standard, uses certifi bundle)
        try:
            import requests

            resp = requests.get(UCI_DOWNLOAD_URL, stream=True, timeout=60)
            resp.raise_for_status()
            with open(zip_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            download_success = True
        except Exception as e:
            print(f"[FlowCast Warning] requests download failed ({e}), trying urllib with certifi...")

        # Method 2: urllib with SSL context fallback
        if not download_success:
            import ssl

            try:
                import certifi

                ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ctx = ssl._create_unverified_context()

            with urllib.request.urlopen(UCI_DOWNLOAD_URL, context=ctx) as u_resp, open(zip_file, "wb") as f:
                f.write(u_resp.read())

    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(raw_dir)

    if not mat_file.exists():
        raise FileNotFoundError(
            f"Expected {mat_file} to exist after extracting {zip_file}"
        )

    return mat_file


def load_raw_traffic_data(
    raw_dir: Path | str = "data/raw",
) -> Dict[str, Any]:
    """Load raw MATLAB matrices into structured NumPy ndarrays.

    Returns:
        Dictionary containing:
            - 'X_tr_raw': ndarray of shape (1261, 36, 48), float64
            - 'X_te_raw': ndarray of shape (840, 36, 48), float64
            - 'Y_tr_raw': ndarray of shape (1261, 36), float64
            - 'Y_te_raw': ndarray of shape (840, 36), float64
            - 'adj_matrix': ndarray of shape (36, 36), uint8
            - 'metadata': Dictionary with dataset citations, shapes, and descriptions
    """
    mat_path = download_dataset_if_needed(raw_dir)
    mat = sio.loadmat(str(mat_path))

    # tra_X_tr has shape (1, 1261), each element is a sparse (36, 48) matrix
    tra_X_tr = mat["tra_X_tr"]
    num_tr = tra_X_tr.shape[1]
    X_tr_list = [tra_X_tr[0, t].toarray() for t in range(num_tr)]
    X_tr_raw = np.stack(X_tr_list, axis=0).astype(np.float32)

    # tra_X_te has shape (1, 840), each element is a sparse (36, 48) matrix
    tra_X_te = mat["tra_X_te"]
    num_te = tra_X_te.shape[1]
    X_te_list = [tra_X_te[0, t].toarray() for t in range(num_te)]
    X_te_raw = np.stack(X_te_list, axis=0).astype(np.float32)

    # tra_Y_tr has shape (36, 1261) -> transpose to (1261, 36)
    Y_tr_raw = mat["tra_Y_tr"].T.astype(np.float32)

    # tra_Y_te has shape (36, 840) -> transpose to (840, 36)
    Y_te_raw = mat["tra_Y_te"].T.astype(np.float32)

    # Adjacency matrix has shape (36, 36)
    adj_matrix = mat["tra_adj_mat"].astype(np.uint8)

    metadata = {
        "dataset_name": "Traffic Flow Forecasting",
        "uci_id": 734,
        "doi": "10.1145/3339823",
        "citation": "Liang Zhao, Olga Gkountouna, and Dieter Pfoser. 2019. ACM TSAS 5(3).",
        "license": "CC BY 4.0",
        "num_sensors": 36,
        "features_per_sensor": 48,
        "temporal_history_length": 10,
        "sampling_rate_minutes": 15,
        "train_time_steps": num_tr,
        "test_time_steps": num_te,
        "total_time_steps": num_tr + num_te,
    }

    return {
        "X_tr_raw": X_tr_raw,
        "X_te_raw": X_te_raw,
        "Y_tr_raw": Y_tr_raw,
        "Y_te_raw": Y_te_raw,
        "adj_matrix": adj_matrix,
        "metadata": metadata,
    }
