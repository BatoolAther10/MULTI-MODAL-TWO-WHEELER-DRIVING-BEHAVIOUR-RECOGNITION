"""Train the GRU and LSTM recurrent sequence classifiers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windowing import make_windows


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_data(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load windows from the clean CSV and encode their labels."""
    df = pd.read_csv(ROOT / config["data"]["clean_csv"], encoding="utf-8-sig")
    X, y, _ = make_windows(
        df,
        window_size=int(config["runtime"]["window_size"]),
        stride=int(config["runtime"]["stride"]),
    )
    labels = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    return X.astype(np.float32), np.asarray([label_to_index[label] for label in y], dtype=np.int32), labels


def build_model(name: str, class_count: int) -> tf.keras.Model:
    """Build the requested recurrent classifier architecture."""
    inputs = tf.keras.Input(shape=(156, 7), dtype=tf.float32)
    if name == "gru":
        hidden = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
        hidden = tf.keras.layers.Dropout(0.2)(hidden)
        hidden = tf.keras.layers.GRU(32)(hidden)
    elif name == "lstm":
        hidden = tf.keras.layers.LSTM(64)(inputs)
        hidden = tf.keras.layers.Dropout(0.2)(hidden)
    else:
        raise ValueError(f"Unsupported recurrent model: {name}")
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(hidden)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    """Export a Keras recurrent model as float16 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    output_path.write_bytes(converter.convert())


def train_one(name: str, X: np.ndarray, y: np.ndarray, labels: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """Train, evaluate, and export one recurrent classifier."""
    seed = int(config["experiment"]["seed"])
    tf.keras.utils.set_random_seed(seed)
    model = build_model(name, len(labels))
    model.fit(
        X,
        y,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )
    predictions = np.argmax(model.predict(X, verbose=0), axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "label_order": labels,
        "n_windows": int(len(X)),
    }
    models_dir = ROOT / config["model"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    keras_path = models_dir / f"{name}.keras"
    tflite_path = models_dir / f"{name}.tflite"
    model.save(keras_path)
    export_tflite(model, tflite_path)
    metrics.update({"model_path": str(tflite_path), "fallback_path": str(keras_path)})
    log_path = ROOT / "logs" / f"{name}_recurrent_metrics.json"
    log_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"{name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def train_recurrent() -> dict[str, dict[str, Any]]:
    """Train both requested recurrent models."""
    config = load_config()
    X, y, labels = load_data(config)
    return {name: train_one(name, X, y, labels, config) for name in ("gru", "lstm")}


if __name__ == "__main__":
    train_recurrent()
