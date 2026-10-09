"""Benchmark the registered model adapters on a fixed number of windows."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_layer.gru_adapter import GRUAdapter
from inference_layer.model_factory import create_model
from inference_layer.resnet1d_adapter import ResNet1DAdapter
from inference_layer.s5_adapter import S5Adapter
from inference_layer.tcnca_adapter import TCNCAAdapter
from inference_layer.tst_adapter import TSTAdapter
from inference_layer.xgboost_adapter import XGBoostAdapter
from scripts.train_gru import train_gru
from scripts.train_resnet1d import train_resnet1d
from scripts.train_s5 import train_s5
from scripts.train_tcnca import train_tcnca
from scripts.train_tst import train_tst
from scripts.train_xgboost import train_xgboost
from utils.windowing import make_windows


def benchmark_models(window_count: int = 500, csv_path: str = "logs/benchmark.csv") -> pd.DataFrame:
    """Benchmark all registered adapters, returning a dataframe of accuracy/F1/latency."""
    df = pd.read_csv(ROOT / "data/processed/clean.csv", encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    X = X[:window_count]
    y = y[:window_count]

    class_order = sorted(np.unique(y).tolist())
    label_to_idx = {label: idx for idx, label in enumerate(class_order)}
    y_idx = np.asarray([label_to_idx[label] for label in y], dtype=np.int64)

    xgb_model_path = ROOT / "models" / "xgboost.json"
    if not xgb_model_path.exists():
        train_xgboost(str(ROOT / "data/processed/clean.csv"), str(xgb_model_path))

    gru_model_path = ROOT / "models" / "gru.tflite"
    if not gru_model_path.exists():
        train_gru(str(ROOT / "data/processed/clean.csv"), str(gru_model_path))

    resnet_model_path = ROOT / "models" / "resnet1d.onnx"
    if not resnet_model_path.exists():
        train_resnet1d(str(ROOT / "data/processed/clean.csv"), str(resnet_model_path))

    tst_model_path = ROOT / "models" / "tst.tflite"
    if not tst_model_path.exists():
        train_tst(str(ROOT / "data/processed/clean.csv"), str(tst_model_path))

    tcnca_model_path = ROOT / "models" / "tcnca.tflite"
    if not tcnca_model_path.exists():
        train_tcnca(str(ROOT / "data/processed/clean.csv"), str(tcnca_model_path))

    s5_model_path = ROOT / "models" / "s5.tflite"
    if not s5_model_path.exists():
        train_s5(str(ROOT / "data/processed/clean.csv"), str(s5_model_path))

    rows: list[dict] = []
    for name in ["stub", "xgboost", "gru", "resnet1d", "tst", "tcnca", "s5"]:
        model = create_model(name)
        if name == "xgboost":
            model = XGBoostAdapter(num_classes=len(class_order), label_order=class_order)
            model.load(str(xgb_model_path))
        elif name == "gru":
            model = GRUAdapter(model_path=str(gru_model_path), num_classes=len(class_order))
            model.load(str(gru_model_path))
        elif name == "resnet1d":
            model = ResNet1DAdapter(model_path=str(resnet_model_path), num_classes=len(class_order))
            model.load(str(resnet_model_path))
        elif name == "tst":
            model = TSTAdapter(model_path=str(tst_model_path), num_classes=len(class_order))
            model.load(str(tst_model_path))
        elif name == "tcnca":
            model = TCNCAAdapter(model_path=str(tcnca_model_path), num_classes=len(class_order))
            model.load(str(tcnca_model_path))
        elif name == "s5":
            model = S5Adapter(model_path=str(s5_model_path), num_classes=len(class_order))
            model.load(str(s5_model_path))

        model_predictions: list[int] = []
        latencies: list[float] = []
        for window in X:
            start = time.perf_counter()
            probs = model.predict(window)
            elapsed = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed)
            pred_idx = int(np.argmax(probs))
            model_predictions.append(pred_idx)

        acc = accuracy_score(y_idx, model_predictions)
        f1 = f1_score(y_idx, model_predictions, average="macro")
        rows.append({
            "model": name,
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "latency_ms": float(np.median(latencies)),
            "num_params": 0,
        })

    output = ROOT / csv_path
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df = pd.DataFrame(rows)
    benchmark_df.to_csv(output, index=False)
    print(benchmark_df.to_string(index=False))
    return benchmark_df


if __name__ == "__main__":
    benchmark_models()
