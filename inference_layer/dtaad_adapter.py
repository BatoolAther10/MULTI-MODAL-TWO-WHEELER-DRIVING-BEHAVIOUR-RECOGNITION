"""TFLite adapter for DTAAD reconstruction-based novelty scoring."""

from __future__ import annotations

import os
import time

import numpy as np
import tensorflow as tf

from inference_layer.base_adapter import BaseOnDeviceModel


class DTAADAdapter(BaseOnDeviceModel):
    """Adapter returning reconstruction error as a one-element novelty score."""

    def __init__(self, model_path: str = "models/dtaad.tflite", num_classes: int = 1) -> None:
        """Create a DTAAD adapter."""
        self.model_path = model_path
        self.num_classes = num_classes
        self.interpreter: tf.lite.Interpreter | None = None
        self.keras_model: tf.keras.Model | None = None
        self._latency_ms = 0.0

    def load(self, path: str) -> None:
        """Load TFLite first and fall back to the adjacent Keras model."""
        self.model_path = path
        keras_path = path.replace(".tflite", ".keras")
        if os.path.exists(path):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=path, num_threads=1)
                self.interpreter.allocate_tensors()
                return
            except (RuntimeError, ValueError):
                self.interpreter = None
        if os.path.exists(keras_path):
            self.keras_model = tf.keras.models.load_model(keras_path)
            return
        raise FileNotFoundError(f"No valid DTAAD model found for path: {path}")

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Return mean squared reconstruction error for one window."""
        if window.shape != (156, 7):
            raise ValueError(f"Expected window shape (156, 7), got {window.shape}.")
        tensor = np.asarray(window, dtype=np.float32).reshape(1, 156, 7)
        start = time.perf_counter_ns()
        if self.interpreter is not None:
            input_details = self.interpreter.get_input_details()[0]
            output_details = self.interpreter.get_output_details()[0]
            self.interpreter.set_tensor(input_details["index"], tensor)
            self.interpreter.invoke()
            reconstruction = self.interpreter.get_tensor(output_details["index"])
        elif self.keras_model is not None:
            reconstruction = self.keras_model(tensor, training=False).numpy()
        else:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        self._latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        error = float(np.mean(np.square(tensor - reconstruction)))
        return np.asarray([error], dtype=np.float32)

    @property
    def latency_ms(self) -> float:
        """Return the latest DTAAD inference latency in milliseconds."""
        return self._latency_ms
