"""Stub adapter used for pipeline validation and fallback bootstrapping."""

from __future__ import annotations

import logging

import numpy as np

from inference_layer.base_adapter import BaseOnDeviceModel


class StubAdapter(BaseOnDeviceModel):
    """Minimal model stub that returns deterministic probabilities."""

    def __init__(self, num_classes: int = 5) -> None:
        """Create a stub adapter with a fixed class count."""
        self.num_classes = num_classes
        self._latency_ms = 0.5

    def load(self, path: str) -> None:
        """Load a model from disk; stub implementation accepts any path."""
        self._loaded_path = path

    def predict(self, window: np.ndarray) -> np.ndarray:
        """Return a softmax-like probability vector for a dummy input window."""
        if window.shape != (156, 7):
            raise ValueError(f"Expected window of shape (156, 7), got {window.shape}.")
        probs = np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)
        return probs

    @property
    def latency_ms(self) -> float:
        """Return the latency estimate for the stub model."""
        return self._latency_ms


class EgoDriveRTAdapter(StubAdapter):
    """Stub for EgoDriveRT because pretrained weights are not public."""

    reason = "NOT_IMPLEMENTED: EgoDriveRT weights are not public; authors were contacted."

    def load(self, path: str) -> None:
        """Record the unavailable EgoDriveRT implementation status."""
        logging.getLogger(__name__).warning(self.reason)
        super().load(path)


class EgoDriveMaxAdapter(StubAdapter):
    """Rejected stub for EgoDriveMax due to deployment constraints."""

    reason = "NOT_IMPLEMENTED: EgoDriveMax rejected because it exceeds 10M parameters and the 10 ms budget."

    def load(self, path: str) -> None:
        """Record the rejected EgoDriveMax implementation status."""
        logging.getLogger(__name__).warning(self.reason)
        super().load(path)
