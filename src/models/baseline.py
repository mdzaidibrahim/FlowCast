"""Conventional Machine Learning Baseline Model for FlowCast.

Implements multi-output Random Forest regression predicting next-15-minute
traffic volume across all 36 sensor stations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class FlowCastBaseline:
    """Random Forest multi-output regressor for benchmark comparison."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> FlowCastBaseline:
        """Fit the multi-output Random Forest regressor on training features.

        Args:
            X: Array of shape (N, 1728) representing flattened historical & contextual features.
            y: Array of shape (N, 36) representing target traffic volumes.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-15m traffic volume across all 36 sensors.

        Args:
            X: Array of shape (N, 1728) or (1728,).

        Returns:
            Array of shape (N, 36) with clipped predictions in [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("FlowCastBaseline must be fitted before predict() is called.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.0)

    def save(self, filepath: Path | str) -> Path:
        """Serialize model weights to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(filepath))
        return filepath

    @classmethod
    def load(cls, filepath: Path | str) -> FlowCastBaseline:
        """Deserialize model from disk."""
        instance = cls()
        instance.model = joblib.load(str(filepath))
        instance.is_fitted = True
        return instance
