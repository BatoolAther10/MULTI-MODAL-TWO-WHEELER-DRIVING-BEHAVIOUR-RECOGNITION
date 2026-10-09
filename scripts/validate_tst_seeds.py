"""Validate leading TST configurations across multiple random seeds."""

from __future__ import annotations

import json
import sys
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
    {"name": "wider", "d_model": 24, "num_heads": 2, "feedforward_dim": 48, "num_blocks": 2, "dropout": 0.1},
    {"name": "more_heads", "d_model": 16, "num_heads": 4, "feedforward_dim": 32, "num_blocks": 2, "dropout": 0.1},
    {"name": "baseline", "d_model": 16, "num_heads": 2, "feedforward_dim": 32, "num_blocks": 2, "dropout": 0.0},
]
SEEDS = [7, 42, 123]


def validate_tst(epochs: int = 10) -> list[dict]:
    """Evaluate leading TST variants on repeated stratified splits."""
    df = pd.read_csv(ROOT / "data/processed/clean.csv", encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    labels = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int32)
    rows = []

    for config in CONFIGS:
        for seed in SEEDS:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            X_train, X_test, y_train, y_test = train_test_split(
                X.astype(np.float32), y_int, test_size=0.2, random_state=seed, stratify=y_int
            )
            model = build_tst(num_classes=len(labels), **{key: value for key, value in config.items() if key != "name"})
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=32, verbose=0)
            predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
            row = {
                **config,
                "seed": seed,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro")),
                "num_params": int(model.count_params()),
            }
            rows.append(row)
            print(row)

    summary = []
    for config in CONFIGS:
        values = [row for row in rows if row["name"] == config["name"]]
        summary.append({
            "name": config["name"],
            "mean_accuracy": float(np.mean([row["accuracy"] for row in values])),
            "std_accuracy": float(np.std([row["accuracy"] for row in values])),
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in values])),
            "std_macro_f1": float(np.std([row["macro_f1"] for row in values])),
            "num_params": values[0]["num_params"],
        })
    summary.sort(key=lambda row: (row["mean_macro_f1"], row["mean_accuracy"]), reverse=True)
    output = ROOT / "logs/tst_seed_validation.json"
    output.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2), encoding="utf-8")
    print("Summary:")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved validation results to: {output}")
    return summary


if __name__ == "__main__":
    validate_tst()
