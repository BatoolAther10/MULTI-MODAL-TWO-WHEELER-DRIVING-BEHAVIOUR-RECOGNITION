"""TFLite S5 adapter with a Keras fallback."""

from __future__ import annotations

import os
import time

import numpy as np
import tensorflow as tf

from inference_layer.base_adapter import BaseOnDeviceModel
from scripts.train_s5 import S5Cell


class S5Adapter(BaseOnDeviceModel):
    """Adapter for the compact S5-style state-space classifier."""

    def __init__(self, model_path: str = "models/s5.tflite", num_classes: int = 5) -> None:
        """Create an S5 adapter."""
        self.model_path = model_path
        self.num_classes = num_classes
        self.interpreter: tf.lite.Interpreter | None = None
        self.keras_model: tf.keras.Model | None = None
        self._keras_predict = None
        self._latency_ms = 0.0

    def load(self, path: str) -> None:
        """Load TFLite first, falling back to the Keras artifact if needed."""
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
            self.keras_model = tf.keras.models.load_model(keras_path, custom_objects={"S5Cell": S5Cell})
            self._keras_predict = tf.function(self.keras_model, reduce_retracing=True)
            return
        raise FileNotFoundError(f"No valid S5 model found for path: {path}")

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Run a single 156x7 window and return class probabilities."""
        if window.shape != (156, 7):
            raise ValueError(f"Expected window shape (156, 7), got {window.shape}.")
        tensor = np.asarray(window, dtype=np.float32).reshape(1, 156, 7)
        start = time.perf_counter()
        if self.interpreter is not None:
            input_details = self.interpreter.get_input_details()[0]
            output_details = self.interpreter.get_output_details()[0]
            self.interpreter.set_tensor(input_details["index"], tensor)
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(output_details["index"])[0]
        elif self._keras_predict is not None:
            probabilities = self._keras_predict(tensor, training=False).numpy()[0]
        else:
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        self._latency_ms = (time.perf_counter() - start) * 1000.0
        return np.asarray(probabilities, dtype=np.float32)

    @property
    def latency_ms(self) -> float:
        """Return the latest inference latency in milliseconds."""
        return self._latency_ms
