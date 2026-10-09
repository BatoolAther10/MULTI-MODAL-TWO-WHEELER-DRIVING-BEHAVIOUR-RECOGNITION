"""Base adapter interface for on-device models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseOnDeviceModel(ABC):
    """Interface for all inference adapters used on device."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a model from disk."""
        pass

    @abstractmethod
    def predict(self, window: np.ndarray) -> np.ndarray:
        """Accept a (156, 7) window and return class probabilities."""
        pass

    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Return the most recent model latency in milliseconds."""
        pass
