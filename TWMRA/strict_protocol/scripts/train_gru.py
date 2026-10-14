"""Train a compact GRU model and export it as TFLite."""

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

from utils.windowing import make_windows


def train_gru(
    csv_path: str = "data/processed/clean.csv",
    model_path: str = "models/gru.tflite",
    logs_path: str = "logs/gru_metrics.json",
) -> dict:
    """Train a compact GRU classifier, export TFLite, and keep a fallback Keras backup."""
    data_path = ROOT / csv_path
    if not data_path.exists():
        raise FileNotFoundError(f"Clean CSV not found: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)

    label_order = sorted(np.unique(y).tolist())
    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_int,
        test_size=0.2,
        random_state=42,
        stratify=y_int,
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(156, 7), dtype=tf.float32),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dense(len(label_order), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=32,
        verbose=0,
    )

    y_proba = model.predict(X_test, verbose=0)
    pred_idx = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_test, pred_idx)
    f1 = f1_score(y_test, pred_idx, average="macro")

    model_output = ROOT / model_path
    model_output.parent.mkdir(parents=True, exist_ok=True)

    fallback_model_path = model_output.with_suffix(".keras")
    model.save(fallback_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    model_output.write_bytes(tflite_model)

    log_output = ROOT / logs_path
    log_output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "label_order": label_order,
        "model_path": str(model_output),
        "fallback_path": str(fallback_model_path),
        "n_windows": int(X.shape[0]),
    }
    log_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"GRU accuracy: {acc:.4f}")
    print(f"GRU macro F1: {f1:.4f}")
    print(f"Saved TFLite model to: {model_output}")
    print(f"Saved Keras fallback to: {fallback_model_path}")
    return metrics


if __name__ == "__main__":
    train_gru()
