"""Run a small offline hyperparameter sweep for the compact TST model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_tst import build_tst
from utils.windowing import make_windows


CONFIGS = [
    {"name": "baseline", "d_model": 16, "num_heads": 2, "feedforward_dim": 32, "num_blocks": 2, "dropout": 0.0, "learning_rate": 1e-3},
    {"name": "wider", "d_model": 24, "num_heads": 2, "feedforward_dim": 48, "num_blocks": 2, "dropout": 0.1, "learning_rate": 1e-3},
    {"name": "deeper", "d_model": 16, "num_heads": 2, "feedforward_dim": 32, "num_blocks": 3, "dropout": 0.1, "learning_rate": 7e-4},
    {"name": "more_heads", "d_model": 16, "num_heads": 4, "feedforward_dim": 32, "num_blocks": 2, "dropout": 0.1, "learning_rate": 7e-4},
    {"name": "regularized", "d_model": 24, "num_heads": 4, "feedforward_dim": 48, "num_blocks": 2, "dropout": 0.2, "learning_rate": 7e-4},
    {"name": "small_fast", "d_model": 12, "num_heads": 2, "feedforward_dim": 24, "num_blocks": 2, "dropout": 0.0, "learning_rate": 1e-3},
]


def sweep_tst(epochs: int = 10) -> list[dict]:
    """Train each TST configuration and save validation metrics."""
    tf.keras.utils.set_random_seed(42)
    df = pd.read_csv(ROOT / "data/processed/clean.csv", encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    label_order = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(label_order)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    results = []
    for config in CONFIGS:
        model = build_tst(
            num_classes=len(label_order),
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            feedforward_dim=config["feedforward_dim"],
            num_blocks=config["num_blocks"],
            dropout=config["dropout"],
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=32, verbose=0)

        probabilities = model.predict(X_test, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        latency_samples = []
        sample = X_test[:1]
        for _ in range(20):
            start = time.perf_counter()
            model(sample, training=False).numpy()
            latency_samples.append((time.perf_counter() - start) * 1000.0)

        results.append({
            **config,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "latency_ms": float(np.median(latency_samples)),
            "num_params": int(model.count_params()),
        })
        print(results[-1])

    results.sort(key=lambda result: (result["macro_f1"], result["accuracy"]), reverse=True)
    output = ROOT / "logs/sweep_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved sweep results to: {output}")
    return results


if __name__ == "__main__":
    sweep_tst()