"""Benchmark the transformer-family models on the common random test split."""

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

from inference_layer.transformer_adapter import TransformerAdapter
from scripts.train_transformers import train_transformers


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def benchmark_transformers() -> pd.DataFrame:
    """Evaluate transformer variants and save benchmark metrics."""
    config = load_config()
    with np.load(ROOT / config["data"]["splits_npz"], allow_pickle=False) as data:
        X_test = np.asarray(data["random_X_test"], dtype=np.float32)
        y_test = np.asarray(data["random_y_test"]).astype(str)
    labels = sorted(np.unique(y_test).tolist())
    mapping = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([mapping[label] for label in y_test], dtype=np.int64)
    names = ["tst", "tcnca_v1", "tcnca_v2", "tcnca_v3", "mega", "fusformer", "conv_transformer"]
    models_dir = ROOT / config["model"]["models_dir"]
    paths = {name: models_dir / f"{name}.tflite" for name in names}
    if not all(path.exists() for path in paths.values()):
        train_transformers()

    rows: list[dict[str, object]] = []
    for name in names:
        adapter = TransformerAdapter(model_path=str(paths[name]), num_classes=len(labels))
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
    output = ROOT / config["logging"].get("benchmark_transformers_csv", "logs/benchmark_transformers.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print("TCNCA final result: v3")
    return result


if __name__ == "__main__":
    benchmark_transformers()
