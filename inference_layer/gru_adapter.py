"""TensorFlow Lite GRU adapter for on-device inference."""

from __future__ import annotations

import os
import time

import numpy as np
import tensorflow as tf

from inference_layer.base_adapter import BaseOnDeviceModel


class GRUAdapter(BaseOnDeviceModel):
    """Adapter wrapping the GRU model with a TFLite-first fallback to the Keras backup."""

    def __init__(self, model_path: str = "models/gru.tflite", num_classes: int = 5) -> None:
        """Create a GRU adapter with the model path fixed at initialization."""
        self.model_path = model_path
        self.num_classes = num_classes
        self.interpreter: tf.lite.Interpreter | None = None
        self.keras_model: tf.keras.Model | None = None
        self._latency_ms = 0.0

    def load(self, path: str) -> None:
        """Load the TFLite interpreter from disk, or fall back to the Keras model if Flex is unavailable."""
        self.model_path = path
        tflite_path = path
        keras_path = path.replace(".tflite", ".keras")

        if os.path.exists(tflite_path):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=1)
                self.interpreter.allocate_tensors()
                return
            except RuntimeError:
                self.interpreter = None

        if os.path.exists(keras_path):
            self.keras_model = tf.keras.models.load_model(keras_path)
            return

        raise FileNotFoundError(f"No valid GRU model found for path: {path}")

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Run a single 156x7 window through the GRU model and return probabilities."""
        if window.shape != (156, 7):
            raise ValueError(f"Expected window shape (156, 7), got {window.shape}.")

        if self.interpreter is not None:
            input_details = self.interpreter.get_input_details()[0]
            output_details = self.interpreter.get_output_details()[0]
            tensor = np.asarray(window, dtype=np.float32).reshape(1, 156, 7)
            start = time.perf_counter()
            self.interpreter.set_tensor(input_details["index"], tensor)
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(output_details["index"])[0].astype(np.float32)
            self._latency_ms = (time.perf_counter() - start) * 1000.0
            return probabilities

        if self.keras_model is not None:
            start = time.perf_counter()
            probabilities = self.keras_model.predict(np.asarray([window], dtype=np.float32), verbose=0)[0].astype(np.float32)
            self._latency_ms = (time.perf_counter() - start) * 1000.0
            return probabilities

        raise RuntimeError("Model is not loaded. Call load() before predict().")

    @property
    def latency_ms(self) -> float:
        """Return the latency of the latest GRU inference in milliseconds."""
        return self._latency_ms
