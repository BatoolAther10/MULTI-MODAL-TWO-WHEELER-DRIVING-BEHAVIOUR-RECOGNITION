"""Measure simulated end-to-end pipeline latency by processing stage."""

from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_layer.model_factory import create_model
from utils.timing import StageTimer

STAGES = [
    "imu_queue_wait_ms",
    "sync_forward_fill_ms",
    "feature_extraction_ms",
    "model_inference_ms",
    "fallback_check_ms",
    "csv_write_buffered_ms",
    "total_e2e_ms",
]


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_windows(config: dict[str, Any]) -> np.ndarray:
    """Load up to 500 real windows for the dummy sensor loop."""
    with np.load(ROOT / config["data"]["windows_npz"], allow_pickle=True) as data:
        windows = np.asarray(data["X"], dtype=np.float32)
    if len(windows) == 0:
        raise ValueError("No processed windows are available for latency benchmarking.")
    return windows


def build_model(config: dict[str, Any]):
    """Load the configured primary model for inference timing."""
    model_name = str(config["model"]["primary"])
    classes = list(config["runtime"]["classes"])
    models_dir = ROOT / config["model"]["models_dir"]
    suffix = ".json" if model_name == "xgboost" else ".tflite"
    optimized_path = models_dir / f"{model_name}_optimized{suffix}"
    path = optimized_path if optimized_path.exists() else models_dir / f"{model_name}{suffix}"
    kwargs: dict[str, object] = {"num_classes": len(classes)}
    if model_name == "xgboost":
        kwargs["label_order"] = classes
    model = create_model(model_name, **kwargs)
    model.load(str(path))
    return model


def simulate_pipeline(window: np.ndarray, model: object, csv_buffer: io.StringIO, queue: Queue[np.ndarray]) -> dict[str, float]:
    """Run one staged dummy pipeline iteration."""
    timings: dict[str, float] = {}
    total_start = time.perf_counter_ns()
    with StageTimer("imu_queue_wait_ms", timings):
        queue.put(window)
        queued_window = queue.get()
    with StageTimer("sync_forward_fill_ms", timings):
        synced_window = np.asarray(queued_window, dtype=np.float32).copy()
        for channel in range(synced_window.shape[1]):
            values = synced_window[:, channel]
            missing = ~np.isfinite(values)
            if missing.any():
                values[missing] = 0.0
    with StageTimer("feature_extraction_ms", timings):
        features = np.ascontiguousarray(synced_window[:, :7], dtype=np.float32)
    with StageTimer("model_inference_ms", timings):
        probabilities = model.predict(features)
    with StageTimer("fallback_check_ms", timings):
        prediction_index = int(np.argmax(probabilities))
        fallback_needed = float(getattr(model, "latency_ms", 0.0) > 10.0)
        prediction_label = str(prediction_index)
    with StageTimer("csv_write_buffered_ms", timings):
        csv_buffer.write(f"{prediction_label},{fallback_needed}\n")
    timings["total_e2e_ms"] = (time.perf_counter_ns() - total_start) / 1_000_000.0
    return timings


def percentile_rows(samples: list[dict[str, float]]) -> pd.DataFrame:
    """Summarize stage timing samples with mean and percentile statistics."""
    rows: list[dict[str, object]] = []
    for stage in STAGES:
        values = np.asarray([sample[stage] for sample in samples], dtype=np.float64)
        rows.append({
            "stage": stage,
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "iterations": int(len(values)),
        })
    return pd.DataFrame(rows)


def benchmark_e2e(iterations: int = 500) -> pd.DataFrame:
    """Run the staged latency benchmark and save summary statistics."""
    config = load_config()
    windows = load_windows(config)
    model = build_model(config)
    queue: Queue[np.ndarray] = Queue(maxsize=1)
    csv_buffer = io.StringIO()
    csv.writer(csv_buffer).writerow(["prediction", "fallback_needed"])
    samples = [simulate_pipeline(windows[index % len(windows)], model, csv_buffer, queue) for index in range(iterations)]
    result = percentile_rows(samples)
    output = ROOT / config["logging"].get("latency_breakdown_csv", "logs/latency_breakdown.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print("model_inference_ms measures only adapter inference; total_e2e_ms includes every simulated pipeline stage.")
    return result


if __name__ == "__main__":
    benchmark_e2e()
