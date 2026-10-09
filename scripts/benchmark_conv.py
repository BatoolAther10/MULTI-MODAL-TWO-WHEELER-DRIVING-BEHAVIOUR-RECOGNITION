"""Benchmark convolutional models on the common random test split."""

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

from inference_layer.convnet_adapter import ConvNetAdapter
from inference_layer.resnet1d_adapter import ResNet1DAdapter
from inference_layer.sgconv_adapter import SGConvAdapter
from scripts.train_conv import train_conv


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def benchmark_conv() -> pd.DataFrame:
    """Evaluate the convolutional models and save benchmark metrics."""
    config = load_config()
    with np.load(ROOT / config["data"]["splits_npz"], allow_pickle=False) as data:
        X_test = np.asarray(data["random_X_test"], dtype=np.float32)
        y_test = np.asarray(data["random_y_test"]).astype(str)
    labels = sorted(np.unique(y_test).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([label_to_index[label] for label in y_test], dtype=np.int64)
    models_dir = ROOT / config["model"]["models_dir"]
    paths = {"resnet1d": models_dir / "resnet1d.onnx", "sgconv": models_dir / "sgconv.onnx", "convnet_har": models_dir / "convnet_har.tflite"}
    if not all(path.exists() for path in paths.values()):
        train_conv()

    rows: list[dict[str, object]] = []
    adapters = {
        "resnet1d": ResNet1DAdapter(model_path=str(paths["resnet1d"]), num_classes=len(labels)),
        "sgconv": SGConvAdapter(model_path=str(paths["sgconv"]), num_classes=len(labels)),
        "convnet_har": ConvNetAdapter(model_path=str(paths["convnet_har"]), num_classes=len(labels)),
    }
    for name, adapter in adapters.items():
        adapter.load(str(paths[name]))
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
    output = ROOT / config["logging"].get("benchmark_conv_csv", "logs/benchmark_conv.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    benchmark_conv()
