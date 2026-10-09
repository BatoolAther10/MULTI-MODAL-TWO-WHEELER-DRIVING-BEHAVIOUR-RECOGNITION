"""ONNX ResNet1D adapter for on-device inference."""

from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort

from inference_layer.base_adapter import BaseOnDeviceModel


class ResNet1DAdapter(BaseOnDeviceModel):
    """Adapter wrapping an exported ResNet1D ONNX model."""

    def __init__(self, model_path: str = "models/resnet1d.onnx", num_classes: int = 5) -> None:
        """Create a ResNet1D ONNX adapter."""
        self.model_path = model_path
        self.num_classes = num_classes
        self.session: ort.InferenceSession | None = None
        self._latency_ms = 0.0

    def load(self, path: str) -> None:
        """Load the ONNX session from disk."""
        self.model_path = path
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Run a single 156x7 window and return class probabilities."""
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        if window.shape != (156, 7):
            raise ValueError(f"Expected window shape (156, 7), got {window.shape}.")

        input_name = self.session.get_inputs()[0].name
        start = time.perf_counter()
        logits = self.session.run(None, {input_name: window.astype(np.float32).reshape(1, 156, 7)})[0][0]
        self._latency_ms = (time.perf_counter() - start) * 1000.0
        return logits.astype(np.float32)

    @property
    def latency_ms(self) -> float:
        """Return the latest inference latency in milliseconds."""
        return self._latency_ms
