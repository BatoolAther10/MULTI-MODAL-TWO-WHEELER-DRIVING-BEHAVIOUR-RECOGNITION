"""Benchmark Group A tabular models on the common random test split."""

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

from inference_layer.model_factory import create_model
from inference_layer.xgboost_adapter import XGBoostAdapter
from scripts.train_tabular import train_all

MODEL_NAMES = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "histgradientboosting",
    "lightgbm",
    "xgboost",
    "mlp_tabular",
]


def load_config() -> dict[str, Any]:
    """Load the project YAML configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parameter_count(model: object) -> int:
    """Estimate the number of learned scalar parameters in an estimator."""
    if hasattr(model, "get_params"):
        params = model.get_params()
        return int(sum(value.size for value in params.values() if isinstance(value, np.ndarray)))
    if hasattr(model, "count_params"):
        return int(model.count_params())
    return 0


def benchmark_tabular() -> pd.DataFrame:
    """Train missing baselines, evaluate them, and save benchmark metrics."""
    config = load_config()
    split_path = ROOT / config["data"]["splits_npz"]
    with np.load(split_path, allow_pickle=False) as data:
        X_test = np.asarray(data["random_X_test"], dtype=np.float32)
        y_test = np.asarray(data["random_y_test"]).astype(str)
    labels = sorted(np.unique(y_test).tolist())
    label_to_idx = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([label_to_idx[label] for label in y_test], dtype=np.int64)

    models_dir = ROOT / config["model"]["models_dir"]
    expected = [models_dir / f"{name}.pkl" for name in MODEL_NAMES if name != "xgboost" and name != "mlp_tabular"]
    expected += [models_dir / "xgboost_tabular.json", models_dir / "mlp_tabular.keras"]
    if not all(path.exists() for path in expected):
        train_all()

    rows: list[dict[str, object]] = []
    for name in MODEL_NAMES:
        if name == "xgboost":
            model = XGBoostAdapter(num_classes=len(labels), label_order=labels)
            model.load(str(models_dir / "xgboost_tabular.json"))
        else:
            model = create_model(name)
            model.load(str(models_dir / ("mlp_tabular.keras" if name == "mlp_tabular" else f"{name}.pkl")))

        predictions: list[int] = []
        latencies: list[float] = []
        for window in X_test:
            start = time.perf_counter_ns()
            probabilities = model.predict(window)
            latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
            predictions.append(int(np.argmax(probabilities)))
        estimator = getattr(model, "model", None)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
            "latency_ms": float(np.median(latencies)),
            "params": parameter_count(estimator) if estimator is not None else 0,
        })

    output = ROOT / config["logging"].get("benchmark_tabular_csv", "logs/benchmark_tabular.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    benchmark_tabular()
