"""Entry point for the two-wheeler on-device inference pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from runtime_runner import RuntimeRunner


def main() -> None:
    """Run one offline inference using a generated sensor window."""
    root = Path(__file__).resolve().parent
    windows = np.load(root / "data/processed/windows.npz", allow_pickle=True)
    runner = RuntimeRunner()
    result = runner.predict(windows["X"][0].astype(np.float32))
    print(result)


if __name__ == "__main__":
    main()
