"""Train and export a compact Temporal Spatial Transformer classifier."""

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


def build_tst(
    num_classes: int = 5,
    d_model: int = 16,
    num_heads: int = 2,
    feedforward_dim: int = 32,
    num_blocks: int = 2,
    dropout: float = 0.0,
) -> tf.keras.Model:
    """Build a configurable compact transformer encoder for one sensor window."""
    inputs = tf.keras.Input(shape=(156, 7), dtype=tf.float32, name="window")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    x = tf.keras.layers.Dense(d_model, name="input_projection")(inputs)
    positions = tf.range(start=0, limit=156, delta=1)
    position_embedding = tf.keras.layers.Embedding(156, d_model, name="position_embedding")(positions)
    x = x + position_embedding

    for block_index in range(num_blocks):
        attention_input = tf.keras.layers.LayerNormalization(name=f"attention_norm_{block_index}")(x)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f"self_attention_{block_index}",
        )(attention_input, attention_input)
        x = tf.keras.layers.Add(name=f"attention_residual_{block_index}")([x, attention])

        feedforward_input = tf.keras.layers.LayerNormalization(name=f"feedforward_norm_{block_index}")(x)
        feedforward = tf.keras.layers.Dense(feedforward_dim, activation="relu", name=f"feedforward_expand_{block_index}")(feedforward_input)
        feedforward = tf.keras.layers.Dense(d_model, name=f"feedforward_project_{block_index}")(feedforward)
        x = tf.keras.layers.Add(name=f"feedforward_residual_{block_index}")([x, feedforward])

    x = tf.keras.layers.LayerNormalization(name="output_norm")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="tst")


def train_tst(
    csv_path: str = "data/processed/clean.csv",
    model_path: str = "models/tst.tflite",
    logs_path: str = "logs/tst_metrics.json",
) -> dict:
    """Train the TST model and export TFLite plus a Keras fallback."""
    data_path = ROOT / csv_path
    if not data_path.exists():
        raise FileNotFoundError(f"Clean CSV not found: {data_path}")

    tf.keras.utils.set_random_seed(42)
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    label_order = sorted(np.unique(y).tolist())
    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32),
        y_int,
        test_size=0.2,
        random_state=42,
        stratify=y_int,
    )

    model = build_tst(
        num_classes=len(label_order),
        d_model=24,
        num_heads=2,
        feedforward_dim=48,
        num_blocks=2,
        dropout=0.1,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, validation_split=0.1, epochs=12, batch_size=32, verbose=0)

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    output_path = ROOT / model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path = output_path.with_suffix(".keras")
    model.save(fallback_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

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

    print(f"TST parameters: {model.count_params()}")
    print(f"TST accuracy: {accuracy:.4f}")
    print(f"TST macro F1: {macro_f1:.4f}")
    print(f"Saved TFLite model to: {output_path}")
    print(f"Saved Keras fallback to: {fallback_path}")
    return metrics


if __name__ == "__main__":
    train_tst()
