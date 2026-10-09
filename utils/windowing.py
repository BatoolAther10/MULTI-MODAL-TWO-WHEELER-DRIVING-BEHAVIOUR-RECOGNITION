"""Utilities for generating fixed-length windows and labels from the riding dataset."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Speed"]
DEFAULT_LABEL_COLUMN = "Label"


def make_windows(
    df: pd.DataFrame,
    window_size: int = 156,
    stride: int = 78,
    feature_cols: Sequence[str] | None = None,
    label_col: str = DEFAULT_LABEL_COLUMN,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return sliding windows, majority-vote labels, and metadata for model training."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive integers.")
    if len(df) < window_size:
        raise ValueError(f"Data length {len(df)} is smaller than window_size={window_size}.")

    cols = list(feature_cols) if feature_cols is not None else list(DEFAULT_FEATURE_COLUMNS)
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns: {missing_cols}")
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    if "Speed" in cols and "Speed" in df.columns:
        df = df.copy()
        df["Speed"] = df["Speed"].ffill().bfill()

    windows: list[np.ndarray] = []
    labels: list[str] = []
    class_counter = Counter()

    max_start = len(df) - window_size
    for start in range(0, max_start + 1, stride):
        end = start + window_size
        window_df = df.iloc[start:end].copy()
        feature_values = window_df[cols].to_numpy(dtype=np.float32)
        windows.append(feature_values)

        vote = Counter(window_df[label_col].astype(str).tolist())
        major_label = max(vote.items(), key=lambda item: (item[1], -sorted(vote).index(item[0])))[0]
        labels.append(major_label)
        class_counter[major_label] += 1

    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=object)
    meta = {
        "window_size": int(window_size),
        "stride": int(stride),
        "feature_cols": list(cols),
        "label_col": label_col,
        "n_windows": int(X.shape[0]),
        "class_balance": dict(sorted(class_counter.items())),
    }
    return X, y, meta


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate fixed-size windows from the riding dataset.")
    parser.add_argument("--csv", type=str, default="data/processed/clean.csv")
    parser.add_argument("--output", type=str, default="data/processed/windows.npz")
    parser.add_argument("--window-size", type=int, default=156)
    parser.add_argument("--stride", type=int, default=78)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    X, y, meta = make_windows(df, window_size=args.window_size, stride=args.stride)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=X, y=y, meta=np.array([meta], dtype=object))
    print(f"Saved windows to {output_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"class balance: {meta['class_balance']}")
