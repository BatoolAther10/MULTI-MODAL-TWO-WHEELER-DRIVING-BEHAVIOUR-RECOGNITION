"""Train and export the compact transformer-family model variants."""

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
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_tst import build_tst
from utils.windowing import make_windows


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_data(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load windows and encode labels."""
    df = pd.read_csv(ROOT / config["data"]["clean_csv"], encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    labels = sorted(np.unique(y).tolist())
    mapping = {label: index for index, label in enumerate(labels)}
    return X.astype(np.float32), np.asarray([mapping[label] for label in y], dtype=np.int32), labels


def attention_block(x: tf.Tensor, d_model: int, heads: int, name: str) -> tf.Tensor:
    """Apply one compact transformer encoder block."""
    normalized = tf.keras.layers.LayerNormalization(name=f"{name}_norm1")(x)
    attention = tf.keras.layers.MultiHeadAttention(heads, d_model // heads, name=f"{name}_attention")(normalized, normalized)
    x = tf.keras.layers.Add(name=f"{name}_residual1")([x, attention])
    normalized = tf.keras.layers.LayerNormalization(name=f"{name}_norm2")(x)
    feedforward = tf.keras.layers.Dense(d_model * 2, activation="gelu", name=f"{name}_expand")(normalized)
    feedforward = tf.keras.layers.Dense(d_model, name=f"{name}_project")(feedforward)
    return tf.keras.layers.Add(name=f"{name}_residual2")([x, feedforward])


def build_tcnca_variant(variant: str, class_count: int, chunk_size: int | None = None, dilations: list[int] | None = None) -> tf.keras.Model:
    """Build one TCNCA ablation variant."""
    inputs = tf.keras.Input((156, 7), name="window")
    channels = 24
    x = tf.keras.layers.Conv1D(channels, 3, padding="same", activation="relu")(inputs)
    for index, dilation in enumerate(dilations or [1, 2, 4, 8, 16]):
        residual = x
        x = tf.keras.layers.Conv1D(channels, 3, padding="same", dilation_rate=dilation, activation="relu", name=f"dilated_{index}")(x)
        x = tf.keras.layers.Add()([x, residual])
    if variant != "v1":
        if chunk_size is None or 156 % chunk_size != 0:
            raise ValueError("chunk_size must divide 156 for chunked attention.")
        chunks = tf.keras.layers.Reshape((156 // chunk_size, chunk_size * channels))(x)
        chunks = tf.keras.layers.Dense(channels)(chunks)
        chunks = attention_block(chunks, channels, 2, f"chunk_attention_{variant}")
        x = tf.keras.layers.UpSampling1D(size=chunk_size)(chunks)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(class_count, activation="softmax")(x), name=f"tcnca_{variant}")


def build_mega(class_count: int) -> tf.keras.Model:
    """Build a moving-average gated attention approximation."""
    inputs = tf.keras.Input((156, 7), name="window")
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(inputs)
    moving = tf.keras.layers.AveragePooling1D(9, strides=1, padding="same")(x)
    gate = tf.keras.layers.Activation("sigmoid")(tf.keras.layers.Conv1D(32, 1)(x))
    x = tf.keras.layers.Add()([tf.keras.layers.Multiply()([gate, x]), moving])
    x = attention_block(x, 32, 4, "mega")
    return tf.keras.Model(inputs, tf.keras.layers.Dense(class_count, activation="softmax")(tf.keras.layers.GlobalAveragePooling1D()(x)), name="mega")


def build_fusformer(class_count: int) -> tf.keras.Model:
    """Build a compact convolutional fusion transformer."""
    inputs = tf.keras.Input((156, 7), name="window")
    branch_a = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(inputs)
    branch_b = tf.keras.layers.Conv1D(16, 11, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Concatenate()([branch_a, branch_b])
    x = attention_block(x, 32, 4, "fusformer")
    return tf.keras.Model(inputs, tf.keras.layers.Dense(class_count, activation="softmax")(tf.keras.layers.GlobalAveragePooling1D()(x)), name="fusformer")


def build_conv_transformer(class_count: int) -> tf.keras.Model:
    """Build a convolutional stem followed by two transformer blocks."""
    inputs = tf.keras.Input((156, 7), name="window")
    x = tf.keras.layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = attention_block(x, 32, 4, "conv_transformer_0")
    x = attention_block(x, 32, 4, "conv_transformer_1")
    return tf.keras.Model(inputs, tf.keras.layers.Dense(class_count, activation="softmax")(tf.keras.layers.GlobalAveragePooling1D()(x)), name="conv_transformer")


def export_tflite(model: tf.keras.Model, path: Path) -> None:
    """Export a transformer model as float16 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    path.write_bytes(converter.convert())


def train_one(name: str, model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, labels: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """Train, export, and score one transformer variant."""
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_split=0.1, epochs=8, batch_size=64, verbose=0)
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    path = ROOT / config["model"]["models_dir"] / f"{name}.tflite"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path.with_suffix(".keras"))
    export_tflite(model, path)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "label_order": labels,
        "model_path": str(path),
        "params": int(model.count_params()),
    }
    (ROOT / "logs" / f"{name}_transformer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"{name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, params={metrics['params']}")
    return metrics


def train_transformers() -> dict[str, dict[str, Any]]:
    """Train TST, TCNCA ablations, MEGA, FusFormer, and ConvTransformer."""
    config = load_config()
    seed = int(config["experiment"]["seed"])
    tf.keras.utils.set_random_seed(seed)
    X, y, labels = load_data(config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    models = {
        "tst": build_tst(len(labels), d_model=24, num_heads=2, feedforward_dim=48, num_blocks=2, dropout=0.1),
        "tcnca_v1": build_tcnca_variant("v1", len(labels), dilations=[1, 2, 4, 8, 16]),
        "tcnca_v2": build_tcnca_variant("v2", len(labels), chunk_size=26, dilations=[1, 2, 4, 8, 16]),
        "tcnca_v3": build_tcnca_variant("v3", len(labels), chunk_size=52, dilations=[1, 2, 4, 8, 16]),
        "mega": build_mega(len(labels)),
        "fusformer": build_fusformer(len(labels)),
        "conv_transformer": build_conv_transformer(len(labels)),
    }
    results = {name: train_one(name, model, X_train, y_train, X_test, y_test, labels, config) for name, model in models.items()}
    return results


if __name__ == "__main__":
    train_transformers()
