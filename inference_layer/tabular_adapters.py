"""Adapters for flattened-window tabular classifiers."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from inference_layer.base_adapter import BaseOnDeviceModel


class TabularAdapter(BaseOnDeviceModel):
    """Adapt a joblib tabular estimator to the window inference interface."""

    def __init__(self, model_path: str | None = None) -> None:
        """Create a tabular adapter with an optional model path."""
        self.model_path = model_path
        self.model: object | None = None
        self.label_order: list[str] = []
        self._latency_ms = 0.0

    def load(self, path: str) -> None:
        """Load an estimator bundle from a joblib artifact."""
        bundle = joblib.load(Path(path))
        if not isinstance(bundle, dict) or "estimator" not in bundle:
            raise ValueError(f"Invalid tabular model bundle: {path}")
        self.model = bundle["estimator"]
        self.label_order = [str(label) for label in bundle["label_order"]]

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Flatten one window and return class probabilities."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        if window.shape != (156, 7):
            raise ValueError(f"Expected window shape (156, 7), got {window.shape}.")
        start = time.perf_counter_ns()
        flat = window.reshape(1, -1).astype(np.float32)
        estimator = self.model
        if hasattr(estimator, "predict_proba"):
            probabilities = np.asarray(estimator.predict_proba(flat)[0], dtype=np.float32)
        else:
            scores = np.asarray(estimator.decision_function(flat)[0], dtype=np.float32)
            scores -= np.max(scores)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities)
        self._latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        return probabilities

    @property
    def latency_ms(self) -> float:
        """Return the latest inference latency in milliseconds."""
        return self._latency_ms


class LogisticRegressionAdapter(TabularAdapter):
    """Adapt the Logistic Regression baseline."""


class LinearSVMAdapter(TabularAdapter):
    """Adapt the linear SVM baseline."""


class RBFSVMAdapter(TabularAdapter):
    """Adapt the RBF-kernel SVM baseline."""


class RandomForestAdapter(TabularAdapter):
    """Adapt the Random Forest baseline."""


class HistGradientBoostingAdapter(TabularAdapter):
    """Adapt the HistGradientBoosting baseline."""


class LightGBMAdapter(TabularAdapter):
    """Adapt the LightGBM baseline."""


class MLPTabularAdapter(BaseOnDeviceModel):
    """Adapt the two-layer Keras tabular classifier."""

    def __init__(self, model_path: str | None = None) -> None:
        """Create a Keras tabular adapter with an optional model path."""
        self.model_path = model_path
        self.model: object | None = None
        self._latency_ms = 0.0

    def load(self, path: str) -> None:
        """Load the Keras model from disk."""
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow is required to load MLP_tabular.") from exc
        self.model = tf.keras.models.load_model(path)

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Flatten one window and return Keras class probabilities."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        if window.shape != (156, 7):
            raise ValueError(f"Expected window shape (156, 7), got {window.shape}.")
        start = time.perf_counter_ns()
        probabilities = np.asarray(self.model.predict(window.reshape(1, -1), verbose=0)[0], dtype=np.float32)
        self._latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        return probabilities

    @property
    def latency_ms(self) -> float:
        """Return the latest inference latency in milliseconds."""
        return self._latency_ms
