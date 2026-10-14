"""Train and export a compact Temporal Convolutional Network with channel attention."""

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


def channel_attention(x: tf.Tensor, channels: int, reduction: int = 4, name: str = "channel_attention") -> tf.Tensor:
    """Apply squeeze-excitation channel attention to a temporal feature map."""
    pooled = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    hidden = max(channels // reduction, 4)
    weights = tf.keras.layers.Dense(hidden, activation="relu", name=f"{name}_reduce")(pooled)
    weights = tf.keras.layers.Dense(channels, activation="sigmoid", name=f"{name}_expand")(weights)
    weights = tf.keras.layers.Reshape((1, channels), name=f"{name}_reshape")(weights)
    return tf.keras.layers.Multiply(name=f"{name}_scale")([x, weights])


def build_tcnca(num_classes: int = 5, channels: int = 16, blocks: int = 2) -> tf.keras.Model:
    """Build a compact dilated temporal convolution model with channel attention."""
    inputs = tf.keras.Input(shape=(156, 7), dtype=tf.float32, name="window")
    x = tf.keras.layers.Conv1D(channels, 3, padding="same", name="input_projection")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    for index in range(blocks):
        residual = x
        dilation = 2**index
        x = tf.keras.layers.Conv1D(channels, 3, padding="same", dilation_rate=dilation, name=f"tcn_conv_{index}")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv1D(channels, 3, padding="same", dilation_rate=dilation, name=f"tcn_refine_{index}")(x)
        x = channel_attention(x, channels, name=f"ca_{index}")
        x = tf.keras.layers.Add(name=f"tcn_residual_{index}")([x, residual])
        x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="tcnca")


def train_tcnca(
    csv_path: str = "data/processed/clean.csv",
    model_path: str = "models/tcnca.tflite",
    logs_path: str = "logs/tcnca_metrics.json",
) -> dict:
    """Train TCNCA, export TFLite, and save a Keras fallback."""
    data_path = ROOT / csv_path
    if not data_path.exists():
        raise FileNotFoundError(f"Clean CSV not found: {data_path}")

    tf.keras.utils.set_random_seed(42)
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    label_order = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(label_order)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    model = build_tcnca(num_classes=len(label_order))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, validation_split=0.1, epochs=12, batch_size=32, verbose=0)

    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    output_path = ROOT / model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path = output_path.with_suffix(".keras")
    model.save(fallback_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    output_path.write_bytes(converter.convert())

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "label_order": label_order,
        "model_path": str(output_path),
        "fallback_path": str(fallback_path),
        "num_params": int(model.count_params()),
        "n_windows": int(X.shape[0]),
    }
    log_output = ROOT / logs_path
    log_output.parent.mkdir(parents=True, exist_ok=True)
    log_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"TCNCA parameters: {model.count_params()}")
    print(f"TCNCA accuracy: {accuracy:.4f}")
    print(f"TCNCA macro F1: {macro_f1:.4f}")
    print(f"Saved TFLite model to: {output_path}")
    return metrics


if __name__ == "__main__":
    train_tcnca()
