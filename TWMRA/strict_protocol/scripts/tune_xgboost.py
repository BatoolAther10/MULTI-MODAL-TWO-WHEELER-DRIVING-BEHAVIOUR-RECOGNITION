"""Tune XGBoost separately and plot hyperparameter variation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windowing import make_windows

CONFIGS = [
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9},
]


def tune_xgboost() -> pd.DataFrame:
    """Train each XGBoost configuration and save metrics and plots."""
    df = pd.read_csv(ROOT / "data/processed/clean.csv", encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    labels = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
    X_flat = X.reshape(len(X), -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    rows = []
    for config in CONFIGS:
        model = XGBClassifier(
            objective="multi:softmax",
            num_class=len(labels),
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            **config,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test).astype(np.int64)
        timings = []
        for window in X_test[:20]:
            start = time.perf_counter()
            model.predict(window.reshape(1, -1))
            timings.append((time.perf_counter() - start) * 1000.0)
        row = {
            **config,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "mae": float(np.mean(np.abs(y_test - predictions))),
            "latency_ms": float(np.median(timings)),
        }
        rows.append(row)
        print(row)

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    output = ROOT / "logs/xgboost_tuning.csv"
    results.to_csv(output, index=False)
    summary = results.iloc[0].to_dict()
    (ROOT / "logs/xgboost_tuning_best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_values = results["max_depth"].astype(str) + " / " + results["n_estimators"].astype(str)
    axes[0].plot(x_values, results["accuracy"], marker="o")
    axes[0].set_title("XGBoost hyperparameters vs accuracy")
    axes[0].set_xlabel("max_depth / n_estimators")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x_values, results["mae"], marker="o", color="#c44e52")
    axes[1].set_title("XGBoost hyperparameters vs MAE")
    axes[1].set_xlabel("max_depth / n_estimators")
    axes[1].set_ylabel("Encoded-label MAE")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure_path = ROOT / "logs/figures/tuning/xgboost_tuning.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(results.to_string(index=False))
    print(f"Saved results to: {output}")
    print(f"Saved best configuration to: {ROOT / 'logs/xgboost_tuning_best.json'}")
    print(f"Saved plot to: {figure_path}")
    return results


if __name__ == "__main__":
    tune_xgboost()
