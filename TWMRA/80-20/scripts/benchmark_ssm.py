"""Benchmark ONNX state-space models on the common random test split."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_layer.state_space_adapter import LRUAdapter, S4DAdapter, S5ONNXAdapter
from scripts.train_ssm import train_ssm


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def benchmark_ssm() -> pd.DataFrame:
    """Evaluate S4D, S5, and LRU ONNX models and save metrics."""
    config = load_config()
    with np.load(ROOT / config["data"]["splits_npz"], allow_pickle=False) as data:
        X_test = np.asarray(data["random_X_test"], dtype=np.float32)
        y_test = np.asarray(data["random_y_test"]).astype(str)
    labels = sorted(np.unique(y_test).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([label_to_index[label] for label in y_test], dtype=np.int64)
    models_dir = ROOT / config["model"]["models_dir"]
    paths = {"s4d": models_dir / "s4d.onnx", "s5_onnx": models_dir / "s5_onnx.onnx", "lru": models_dir / "lru.onnx"}
    if not all(path.exists() for path in paths.values()):
        train_ssm()
    adapters = {
        "s4d": S4DAdapter(model_path=str(paths["s4d"]), num_classes=len(labels)),
        "s5_onnx": S5ONNXAdapter(model_path=str(paths["s5_onnx"]), num_classes=len(labels)),
        "lru": LRUAdapter(model_path=str(paths["lru"]), num_classes=len(labels)),
    }
    rows: list[dict[str, object]] = []
    for name, adapter in adapters.items():
        adapter.load(str(paths[name]))
        predictions: list[int] = []
        latencies: list[float] = []
        for window in X_test:
            start = time.perf_counter_ns()
            predictions.append(int(np.argmax(adapter.predict(window))))
            latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            "latency_ms": float(np.median(latencies)),
            "params": 0,
        })
    result = pd.DataFrame(rows)
    output = ROOT / config["logging"].get("benchmark_ssm_csv", "logs/benchmark_ssm.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    benchmark_ssm()
