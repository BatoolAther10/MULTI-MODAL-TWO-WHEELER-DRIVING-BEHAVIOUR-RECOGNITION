"""XGBoost adapter for on-device inference."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from inference_layer.base_adapter import BaseOnDeviceModel


class XGBoostAdapter(BaseOnDeviceModel):
    """Wrap a trained XGBoost classifier for flattened window inference."""

    def __init__(self, num_classes: int = 5, label_order: list[str] | None = None) -> None:
        """Create an adapter bound to a specific class ordering."""
        self.num_classes = num_classes
        self.label_order = label_order or ["BUMP", "LEFT", "RIGHT", "STOP", "STRAIGHT"]
        self.model: XGBClassifier | None = None
        self._latency_ms = 0.0

    def load(self, path: str) -> None:
        """Load the XGBoost model from its JSON artifact."""
        self.model = XGBClassifier()
        self.model.load_model(path)

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Flatten the 156x7 window and return softmax-style probabilities."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        if window.shape != (156, 7):
            raise ValueError(f"Expected window shape (156, 7), got {window.shape}.")

        flat = window.reshape(1, -1).astype(np.float32)
        start = time.perf_counter()
        pred_idx = int(self.model.predict(flat)[0])
        elapsed = (time.perf_counter() - start) * 1000.0
        self._latency_ms = elapsed

        class_probs = np.zeros(self.num_classes, dtype=np.float32)
        class_probs[pred_idx] = 1.0
        return class_probs

    @property
    def latency_ms(self) -> float:
        """Return the latest inference latency in milliseconds."""
        return self._latency_ms
