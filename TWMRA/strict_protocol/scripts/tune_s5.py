"""Tune S5 state size separately and plot variation."""

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
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_s5 import build_s5
from utils.windowing import make_windows

CONFIGS = [{"state_size": 8}, {"state_size": 16}, {"state_size": 24}]


def tune_s5(epochs: int = 4) -> pd.DataFrame:
    """Train compact S5 candidates and save metrics plus variation plots."""
    df = pd.read_csv(ROOT / "data/processed/clean.csv", encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    labels = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    rows = []
    for config in CONFIGS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = build_s5(num_classes=len(labels), **config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=32, verbose=0)
        predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
        timings = []
        for window in X_test[:3]:
            start = time.perf_counter()
            model(window[None, ...], training=False).numpy()
            timings.append((time.perf_counter() - start) * 1000.0)
        row = {
            **config,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "mae": float(np.mean(np.abs(y_test.astype(np.float32) - predictions.astype(np.float32)))),
            "latency_ms": float(np.median(timings)),
            "num_params": int(model.count_params()),
        }
        rows.append(row)
        print(row)

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    output = ROOT / "logs/s5_tuning.csv"
    results.to_csv(output, index=False)
    (ROOT / "logs/s5_tuning_best.json").write_text(
        json.dumps(results.iloc[0].to_dict(), indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(results["state_size"].astype(str), results["accuracy"], marker="o")
    axes[0].set_title("S5 state size vs accuracy")
    axes[0].set_xlabel("State size")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[1].plot(results["state_size"].astype(str), results["mae"], marker="o", color="#c44e52")
    axes[1].set_title("S5 state size vs MAE")
    axes[1].set_xlabel("State size")
    axes[1].set_ylabel("Encoded-label MAE")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure_path = ROOT / "logs/figures/tuning/s5_tuning.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(results.to_string(index=False))
    print(f"Saved results to: {output}")
    print(f"Saved plot to: {figure_path}")
    return results


if __name__ == "__main__":
    tune_s5()
