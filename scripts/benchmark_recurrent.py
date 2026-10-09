"""Benchmark recurrent classifiers on the shared random test split."""

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

from inference_layer.gru_adapter import GRUAdapter
from inference_layer.lstm_adapter import LSTMAdapter
from scripts.train_recurrent import train_recurrent


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def benchmark_recurrent() -> pd.DataFrame:
    """Evaluate GRU and LSTM and save their benchmark metrics."""
    config = load_config()
    with np.load(ROOT / config["data"]["splits_npz"], allow_pickle=False) as data:
        X_test = np.asarray(data["random_X_test"], dtype=np.float32)
        y_test = np.asarray(data["random_y_test"]).astype(str)
    labels = sorted(np.unique(y_test).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([label_to_index[label] for label in y_test], dtype=np.int64)
    models_dir = ROOT / config["model"]["models_dir"]
    required = [models_dir / "gru.tflite", models_dir / "lstm.tflite"]
    if not all(path.exists() for path in required):
        train_recurrent()

    rows: list[dict[str, object]] = []
    for name, adapter_class in (("gru", GRUAdapter), ("lstm", LSTMAdapter)):
        adapter = adapter_class(model_path=str(models_dir / f"{name}.tflite"), num_classes=len(labels))
        adapter.load(str(models_dir / f"{name}.tflite"))
        predictions: list[int] = []
        latencies: list[float] = []
        for window in X_test:
            start = time.perf_counter_ns()
            probabilities = adapter.predict(window)
            latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
            predictions.append(int(np.argmax(probabilities)))
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            "latency_ms": float(np.median(latencies)),
            "params": 0,
        })
    result = pd.DataFrame(rows)
    output = ROOT / config["logging"].get("benchmark_recurrent_csv", "logs/benchmark_recurrent.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    benchmark_recurrent()
