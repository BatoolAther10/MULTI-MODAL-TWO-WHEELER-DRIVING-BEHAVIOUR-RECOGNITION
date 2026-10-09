"""Smoke-test the model factory and stub adapter using a dummy window."""

from __future__ import annotations

import numpy as np

from inference_layer.model_factory import create_model


def main() -> None:
    """Instantiate a stub adapter and validate baseline inference."""
    model = create_model("stub")
    dummy_window = np.random.randn(156, 7).astype(np.float32)
    out = model.predict(dummy_window)
    print(f"Model type: {type(model).__name__}")
    print(f"Output shape: {out.shape}")
    print(f"Output sum: {out.sum():.6f}")
    print(f"Latency ms: {model.latency_ms}")


if __name__ == "__main__":
    main()
