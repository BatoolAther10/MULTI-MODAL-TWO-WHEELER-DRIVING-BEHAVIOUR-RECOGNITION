"""ONNX adapter shared by compact state-space classifiers."""

from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort

from inference_layer.base_adapter import BaseOnDeviceModel


class StateSpaceAdapter(BaseOnDeviceModel):
    """Adapter for an exported S4D, S5, or LRU ONNX classifier."""

    def __init__(self, model_path: str = "models/s4d.onnx", num_classes: int = 5) -> None:
        """Create a state-space adapter."""
        self.model_path = model_path
        self.num_classes = num_classes
        self.session: ort.InferenceSession | None = None
        self._latency_ms = 0.0

    def load(self, path: str) -> None:
        """Load the ONNX inference session."""
        self.model_path = path
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Run one 156x7 window and return class logits."""
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        if window.shape != (156, 7):
            raise ValueError(f"Expected window shape (156, 7), got {window.shape}.")
        input_name = self.session.get_inputs()[0].name
        start = time.perf_counter_ns()
        logits = self.session.run(None, {input_name: window.astype(np.float32).reshape(1, 156, 7)})[0][0]
        self._latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        return logits.astype(np.float32)

    @property
    def latency_ms(self) -> float:
        """Return the latest state-space inference latency in milliseconds."""
        return self._latency_ms


class S4DAdapter(StateSpaceAdapter):
    """Adapter for the S4D ONNX classifier."""


class S5ONNXAdapter(StateSpaceAdapter):
    """Adapter for the ONNX S5 classifier."""


class LRUAdapter(StateSpaceAdapter):
    """Adapter for the LRU ONNX classifier."""
