"""Offline primary/fallback inference runner for Raspberry Pi deployment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from inference_layer.model_factory import create_model


class RuntimeRunner:
    """Run the configured primary model and fall back when it is unsafe to use."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load configuration and initialize the configured model adapters."""
        self.root = Path(__file__).resolve().parent
        with (self.root / config_path).open("r", encoding="utf-8") as handle:
            self.config: dict[str, Any] = yaml.safe_load(handle)

        model_config = self.config["model"]
        classes = self.config["runtime"]["classes"]
        self.classes = list(classes)
        self.primary_name = str(model_config["primary"])
        self.fallback_name = str(model_config["fallback"])
        self.latency_budget_ms = float(model_config["latency_budget_ms"])
        self.primary = self._load_model(self.primary_name)
        self.fallback = self._load_model(self.fallback_name)

    def _load_model(self, name: str):
        """Create and load one registered model from the configured artifact."""
        models_dir = self.root / self.config["model"]["models_dir"]
        model_config = self.config["model"]
        configured_key = None
        if name == self.primary_name:
            configured_key = "path"
        elif name == self.fallback_name:
            configured_key = "fallback_path"

        configured_path = model_config.get(configured_key) if configured_key else None
        if configured_path:
            model_path = self.root / str(configured_path)
        else:
            suffix = ".json" if name == "xgboost" else ".tflite"
            optimized_path = models_dir / f"{name}_optimized{suffix}"
            model_path = optimized_path if optimized_path.exists() else models_dir / f"{name}{suffix}"

        if not model_path.exists():
            raise FileNotFoundError(f"Configured {name} model artifact not found: {model_path}")

        kwargs = {"num_classes": len(self.classes)}
        if name == "xgboost":
            kwargs["label_order"] = self.classes
        model = create_model(name, **kwargs)
        model.load(str(model_path))
        return model

    def predict(self, window: np.ndarray) -> dict[str, Any]:
        """Predict one sensor window, falling back on failure or budget violation."""
        primary_error: str | None = None
        primary_latency_ms: float | None = None
        try:
            start = time.perf_counter()
            probabilities = self.primary.predict(window)
            primary_latency_ms = (time.perf_counter() - start) * 1000.0
            if primary_latency_ms <= self.latency_budget_ms:
                index = int(np.argmax(probabilities))
                return {
                    "label": self.classes[index],
                    "model": self.primary_name,
                    "latency_ms": primary_latency_ms,
                    "fallback_used": False,
                }
            primary_error = f"latency {primary_latency_ms:.3f} ms exceeded {self.latency_budget_ms:.3f} ms"
        except Exception as exc:  # Runtime boundary must protect the live pipeline.
            primary_error = f"{type(exc).__name__}: {exc}"

        start = time.perf_counter()
        fallback_probabilities = self.fallback.predict(window)
        fallback_latency_ms = (time.perf_counter() - start) * 1000.0
        index = int(np.argmax(fallback_probabilities))
        return {
            "label": self.classes[index],
            "model": self.fallback_name,
            "latency_ms": fallback_latency_ms,
            "fallback_used": True,
            "primary_error": primary_error,
            "primary_latency_ms": primary_latency_ms,
        }
