"""Benchmark every implemented model adapter on one common test subset."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_layer.model_factory import create_model

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "histgradientboosting",
    "lightgbm",
    "xgboost",
    "mlp_tabular",
    "gru",
    "lstm",
    "resnet1d",
    "convnet_har",
    "sgconv",
    "s4d",
    "s5_onnx",
    "lru",
    "tst",
    "tcnca_v1",
    "tcnca_v2",
    "tcnca_v3",
    "mega",
    "fusformer",
    "conv_transformer",
    "egodrive_rt",
    "egodrive_max",
]


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def artifact_path(name: str, models_dir: Path) -> Path | None:
    """Resolve the existing artifact path for one registered model."""
    if name in {"logistic_regression", "linear_svm", "rbf_svm", "random_forest", "histgradientboosting", "lightgbm"}:
        return models_dir / f"{name}.pkl"
    if name == "xgboost":
        return models_dir / "xgboost_tabular.json"
    if name == "mlp_tabular":
        return models_dir / "mlp_tabular.keras"
    if name in {"resnet1d", "sgconv", "s4d", "s5_onnx", "lru"}:
        return models_dir / f"{name}.onnx"
    if name in {"egodrive_rt", "egodrive_max"}:
        return models_dir / "stub"
    return models_dir / f"{name}.tflite"


def parameter_count(name: str, path: Path) -> int:
    """Read a saved parameter count or estimate it from a serialized estimator."""
    metrics_paths = [ROOT / "logs" / f"{name}_transformer_metrics.json", ROOT / "logs" / f"{name}_conv_metrics.json", ROOT / "logs" / f"{name}_metrics.json", ROOT / "logs" / f"{name}_recurrent_metrics.json"]
    for metrics_path in metrics_paths:
        if metrics_path.exists():
            try:
                return int(json.loads(metrics_path.read_text(encoding="utf-8")).get("params", json.loads(metrics_path.read_text(encoding="utf-8")).get("num_params", 0)))
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
    if path.suffix == ".pkl" and path.exists():
        bundle = joblib.load(path)
        estimator = bundle.get("estimator") if isinstance(bundle, dict) else bundle
        return int(sum(value.size for value in estimator.__dict__.values() if isinstance(value, np.ndarray)))
    return 0


def build_adapter(name: str, path: Path, label_count: int, labels: list[str]):
    """Create and load one adapter from its configured artifact."""
    tabular_names = {"logistic_regression", "linear_svm", "rbf_svm", "random_forest", "histgradientboosting", "lightgbm", "mlp_tabular"}
    kwargs: dict[str, object] = {}
    if name not in tabular_names:
        kwargs["num_classes"] = label_count
    if name == "xgboost":
        kwargs["num_classes"] = label_count
        kwargs["label_order"] = labels
    adapter = create_model(name, **kwargs)
    adapter.load(str(path))
    return adapter


def benchmark_model(name: str, adapter: object, X: np.ndarray, y_true: np.ndarray, labels: list[str], artifact: Path) -> dict[str, object]:
    """Benchmark one classification adapter and calculate class-wise F1."""
    predictions: list[int] = []
    latencies: list[float] = []
    for window in X:
        start = time.perf_counter_ns()
        probabilities = adapter.predict(window)
        latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
        predictions.append(int(np.argmax(probabilities)))
    per_class = f1_score(y_true, predictions, labels=np.arange(len(labels)), average=None, zero_division=0)
    return {
        "model": name,
        "status": "benchmarked",
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "per_class_f1": json.dumps({label: float(score) for label, score in zip(labels, per_class)}, sort_keys=True),
        "params": parameter_count(name, artifact),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def anomaly_row(X: np.ndarray, labels: np.ndarray, config: dict[str, Any]) -> dict[str, object]:
    """Benchmark DTAAD separately as an anomaly-only model."""
    from inference_layer.dtaad_adapter import DTAADAdapter

    path = ROOT / config["model"]["models_dir"] / "dtaad.tflite"
    adapter = DTAADAdapter()
    adapter.load(str(path))
    scores: list[float] = []
    latencies: list[float] = []
    for window in X:
        start = time.perf_counter_ns()
        scores.append(float(adapter.predict(window)[0]))
        latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "model": "dtaad",
        "status": "anomaly_only",
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "per_class_f1": "{}",
        "params": parameter_count("dtaad", path),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "mean_novelty_score": float(np.mean(scores)),
    }


def benchmark_all(window_count: int = 500) -> pd.DataFrame:
    """Benchmark all available models and save the unified results table."""
    config = load_config()
    with np.load(ROOT / config["data"]["splits_npz"], allow_pickle=False) as data:
        X = np.asarray(data["random_X_test"], dtype=np.float32)
        y = np.asarray(data["random_y_test"]).astype(str)
    benchmark_source = "random_test"
    if len(X) < window_count:
        with np.load(ROOT / config["data"]["windows_npz"], allow_pickle=True) as data:
            X = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"]).astype(str)
        benchmark_source = "full_windows_fallback"
    X = X[:window_count]
    y = y[:window_count]
    labels = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
    models_dir = ROOT / config["model"]["models_dir"]
    rows: list[dict[str, object]] = []
    for name in CLASSIFICATION_MODELS:
        path = artifact_path(name, models_dir)
        if path is None or (name not in {"egodrive_rt", "egodrive_max"} and not path.exists()):
            rows.append({"model": name, "status": "missing_artifact", "accuracy": np.nan, "macro_f1": np.nan, "per_class_f1": "{}", "params": 0, "latency_mean_ms": np.nan, "latency_p95_ms": np.nan})
            continue
        adapter = build_adapter(name, path, len(labels), labels)
        rows.append(benchmark_model(name, adapter, X, y_true, labels, path))
    dtaad_path = models_dir / "dtaad.tflite"
    if dtaad_path.exists():
        rows.append(anomaly_row(X, y, config))
    result = pd.DataFrame(rows)
    result["benchmark_source"] = benchmark_source
    result["window_count"] = len(X)
    output = ROOT / config["logging"].get("benchmark_all_csv", "logs/benchmark_all.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.sort_values("macro_f1", na_position="last", ascending=False).to_string(index=False))
    return result


if __name__ == "__main__":
    benchmark_all()
