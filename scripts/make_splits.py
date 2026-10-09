"""Create reproducible random and session-wise dataset splits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.splits import random_split, session_wise_split
from utils.windowing import make_windows


def load_config(path: Path) -> dict[str, Any]:
    """Load the project YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def derive_session_ids(df: pd.DataFrame, window_count: int, window_size: int, stride: int, gap_s: float) -> np.ndarray:
    """Assign each generated window to its majority timestamp-gap session."""
    if "Raw_Timestamp" not in df.columns:
        raise KeyError("The source data must contain Raw_Timestamp for session splitting.")

    timestamps = pd.to_numeric(df["Raw_Timestamp"], errors="coerce")
    session_ids = timestamps.diff().gt(gap_s).cumsum().astype(int)
    window_sessions: list[int] = []
    for start in range(0, len(df) - window_size + 1, stride):
        values = session_ids.iloc[start : start + window_size]
        window_sessions.append(int(values.mode().iloc[0]))
    result = np.asarray(window_sessions[:window_count], dtype=np.int64)
    if len(result) != window_count:
        raise ValueError("Could not derive one session identifier per window.")
    return result


def load_windows(npz_path: Path, csv_path: Path, window_size: int, stride: int, gap_s: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load window data and recover session metadata when legacy metadata is unreadable."""
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            X = np.asarray(data["X"])
            y = np.asarray(data["y"])
    except (ModuleNotFoundError, ValueError, OSError, KeyError):
        source = pd.read_csv(csv_path, encoding="utf-8-sig")
        X, y, _ = make_windows(source, window_size=window_size, stride=stride)
    source = pd.read_csv(csv_path, encoding="utf-8-sig")
    session_ids = derive_session_ids(source, len(X), window_size, stride, gap_s)
    meta = pd.DataFrame({"session_id": session_ids})
    return X.astype(np.float32), y.astype(str), meta


def distribution(labels: np.ndarray) -> dict[str, int]:
    """Return sorted class counts for a label array."""
    values, counts = np.unique(labels.astype(str), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def save_split_artifact(output_path: Path, splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """Save all named train/test splits to one NPZ artifact."""
    arrays: dict[str, np.ndarray] = {}
    for name, (X_train, X_test, y_train, y_test) in splits.items():
        arrays[f"{name}_X_train"] = X_train
        arrays[f"{name}_X_test"] = X_test
        arrays[f"{name}_y_train"] = y_train
        arrays[f"{name}_y_test"] = y_test
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    """Create and report all configured dataset splits."""
    config = load_config(ROOT / "config.yaml")
    data_config = config["data"]
    runtime_config = config["runtime"]
    npz_path = ROOT / data_config["windows_npz"]
    csv_path = ROOT / data_config["clean_csv"]
    X, y, meta = load_windows(
        npz_path,
        csv_path,
        int(runtime_config["window_size"]),
        int(runtime_config["stride"]),
        float(data_config.get("session_gap_threshold_s", 60.0)),
    )

    sessions = sorted(meta["session_id"].unique().tolist())
    if len(sessions) < 2:
        raise ValueError(f"Expected at least two sessions, found {sessions}.")
    first_session, second_session = sessions[:2]
    splits = {
        "random": random_split(X, y),
        f"session_{first_session}_test": session_wise_split(X, y, meta, [first_session]),
        f"session_{second_session}_test": session_wise_split(X, y, meta, [second_session]),
    }
    save_split_artifact(ROOT / data_config.get("splits_npz", "data/processed/splits.npz"), splits)

    for name, (X_train, X_test, y_train, y_test) in splits.items():
        print(f"{name}: X_train={X_train.shape}, X_test={X_test.shape}")
        print(f"  train distribution: {distribution(y_train)}")
        print(f"  test distribution:  {distribution(y_test)}")


if __name__ == "__main__":
    main()
