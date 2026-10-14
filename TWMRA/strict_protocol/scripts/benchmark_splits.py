"""Compare every classification adapter across random and session-wise splits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_all import CLASSIFICATION_MODELS, artifact_path, build_adapter, load_config


def load_split_data(config: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray, str, str]]:
    """Load random and both session-held-out test splits."""
    path = ROOT / config["data"]["splits_npz"]
    with np.load(path, allow_pickle=False) as data:
        return {
            "random": (np.asarray(data["random_X_test"], dtype=np.float32), np.asarray(data["random_y_test"]).astype(str), "all", "random_test"),
            "session_0_test": (np.asarray(data["session_0_test_X_test"], dtype=np.float32), np.asarray(data["session_0_test_y_test"]).astype(str), "1", "0"),
            "session_1_test": (np.asarray(data["session_1_test_X_test"], dtype=np.float32), np.asarray(data["session_1_test_y_test"]).astype(str), "0", "1"),
        }


def evaluate_adapter(adapter: object, X: np.ndarray, y: np.ndarray, labels: list[str]) -> tuple[float, float]:
    """Evaluate one loaded adapter and return accuracy and macro F1."""
    mapping = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([mapping[label] for label in y], dtype=np.int64)
    predictions = [int(np.argmax(adapter.predict(window))) for window in X]
    return float(accuracy_score(y_true, predictions)), float(f1_score(y_true, predictions, labels=np.arange(len(labels)), average="macro", zero_division=0))


def benchmark_splits() -> pd.DataFrame:
    """Evaluate all classification adapters across the configured splits."""
    config = load_config()
    splits = load_split_data(config)
    labels = list(config["runtime"]["classes"])
    models_dir = ROOT / config["model"]["models_dir"]
    rows: list[dict[str, object]] = []
    for model_name in CLASSIFICATION_MODELS:
        artifact = artifact_path(model_name, models_dir)
        if artifact is None or (model_name not in {"egodrive_rt", "egodrive_max"} and not artifact.exists()):
            for split_name, (_, _, train_sessions, test_sessions) in splits.items():
                rows.append({"model": model_name, "split_type": split_name, "train_sessions": train_sessions, "test_sessions": test_sessions, "accuracy": np.nan, "macro_f1": np.nan, "status": "missing_artifact"})
            continue
        adapter = build_adapter(model_name, artifact, len(labels), sorted(labels))
        for split_name, (X_test, y_test, train_sessions, test_sessions) in splits.items():
            accuracy, macro_f1 = evaluate_adapter(adapter, X_test, y_test, sorted(labels))
            rows.append({
                "model": model_name,
                "split_type": split_name,
                "train_sessions": train_sessions,
                "test_sessions": test_sessions,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "status": "benchmarked",
                "window_count": len(X_test),
            })

    result = pd.DataFrame(rows)
    output = ROOT / config["logging"].get("benchmark_splits_csv", "logs/benchmark_splits.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    random_scores = result[result["split_type"] == "random"].set_index("model")["accuracy"]
    session_scores = result[result["split_type"].str.startswith("session_")].groupby("model")["accuracy"].mean()
    summary = pd.DataFrame({"random_accuracy": random_scores, "session_wise_accuracy": session_scores})
    summary["drop_points"] = (summary["random_accuracy"] - summary["session_wise_accuracy"]) * 100.0
    summary["generalisability_flag"] = np.where(summary["drop_points"] > 15.0, "not generalisable", "within 15 points")
    print(result.to_string(index=False))
    print("\nAccuracy drop summary:")
    print(summary.sort_values("drop_points", ascending=False).to_string())
    flagged = summary[summary["drop_points"] > 15.0]
    if not flagged.empty:
        print("\nModels flagged as not generalisable (>15 points):", ", ".join(flagged.index.tolist()))
    result.attrs["drop_summary"] = summary
    return result


if __name__ == "__main__":
    benchmark_splits()
