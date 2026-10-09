"""Train/test split helpers for windowed riding data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a reproducible stratified random train/test split."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    labels = np.asarray(y)
    try:
        stratify: np.ndarray | None = labels
        return train_test_split(
            X,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(X, labels, test_size=test_size, random_state=seed)


def session_wise_split(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    test_sessions: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return windows partitioned by the requested test-session identifiers."""
    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError("X, y, and meta must contain the same number of samples.")
    if "session_id" not in meta.columns:
        raise KeyError("meta must contain a 'session_id' column.")
    if not test_sessions:
        raise ValueError("test_sessions must contain at least one session identifier.")

    test_mask = meta["session_id"].isin(test_sessions).to_numpy()
    if not test_mask.any():
        raise ValueError(f"None of the requested test sessions exist: {test_sessions}")
    if test_mask.all():
        raise ValueError("The requested test sessions include every window.")

    return X[~test_mask], X[test_mask], np.asarray(y)[~test_mask], np.asarray(y)[test_mask]
