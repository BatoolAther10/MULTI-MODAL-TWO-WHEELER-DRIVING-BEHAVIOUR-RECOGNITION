"""Tune GRU separately and plot recurrent-model variation."""

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

from utils.windowing import make_windows

CONFIGS = [
    {"gru1": 32, "gru2": 16, "learning_rate": 1e-3},
    {"gru1": 64, "gru2": 32, "learning_rate": 1e-3},
    {"gru1": 64, "gru2": 32, "learning_rate": 5e-4},
]


def build_gru(num_classes: int, config: dict) -> tf.keras.Model:
    """Build a configurable two-layer GRU classifier."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(156, 7), dtype=tf.float32),
            tf.keras.layers.GRU(config["gru1"], return_sequences=True),
            tf.keras.layers.GRU(config["gru2"]),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def tune_gru(epochs: int = 6) -> pd.DataFrame:
    """Train GRU candidates and save metrics plus variation plots."""
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
        model = build_gru(len(labels), config)
        model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=32, verbose=0)
        predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
        timings = []
        for window in X_test[:5]:
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
    output = ROOT / "logs/gru_tuning.csv"
    results.to_csv(output, index=False)
    (ROOT / "logs/gru_tuning_best.json").write_text(
        json.dumps(results.iloc[0].to_dict(), indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_values = results["gru1"].astype(str) + " / " + results["gru2"].astype(str)
    axes[0].plot(x_values, results["accuracy"], marker="o")
    axes[0].set_title("GRU hyperparameters vs accuracy")
    axes[0].set_xlabel("GRU layer widths")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x_values, results["mae"], marker="o", color="#c44e52")
    axes[1].set_title("GRU hyperparameters vs MAE")
    axes[1].set_xlabel("GRU layer widths")
    axes[1].set_ylabel("Encoded-label MAE")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure_path = ROOT / "logs/figures/tuning/gru_tuning.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(results.to_string(index=False))
    print(f"Saved results to: {output}")
    print(f"Saved plot to: {figure_path}")
    return results


if __name__ == "__main__":
    tune_gru()
