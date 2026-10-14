"""Train and export a compact Time Series Transformer classifier."""

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


@tf.keras.utils.register_keras_serializable(package="two_wheeler")
class PositionalEmbedding(tf.keras.layers.Layer):
    """Add learned or sinusoidal positions after projecting sensor channels."""

    def __init__(
        self,
        sequence_length: int,
        d_model: int,
        method: str = "sinusoidal",
        **kwargs: object,
    ) -> None:
        """Create a positional encoding layer."""
        super().__init__(**kwargs)
        if method not in {"embedding", "sinusoidal"}:
            raise ValueError(f"Unknown positional encoding method {method}")
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.method = method
        self.proj = tf.keras.layers.Dense(d_model)
        if method == "embedding":
            self.pos_emb = tf.keras.layers.Embedding(sequence_length, d_model)
        else:
            position = np.arange(sequence_length)[:, np.newaxis]
            div_term = np.exp(
                np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
            )
            encoding = np.zeros((sequence_length, d_model), dtype=np.float32)
            encoding[:, 0::2] = np.sin(position * div_term)
            encoding[:, 1::2] = np.cos(position * div_term[: encoding[:, 1::2].shape[1]])
            self.pe = tf.constant(encoding, dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Project sensor channels and add positional information."""
        projected = self.proj(x)
        if self.method == "embedding":
            positions = tf.range(self.sequence_length)
            return projected + self.pos_emb(positions)
        return projected + self.pe

    def get_config(self) -> dict[str, object]:
        """Return serializable layer configuration."""
        return {
            **super().get_config(),
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
            "method": self.method,
        }


@tf.keras.utils.register_keras_serializable(package="two_wheeler")
class TransformerEncoder(tf.keras.layers.Layer):
    """Apply pre-normalized local self-attention and a feed-forward residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        attn_type: str = "sliding_window",
        window_size: int = 7,
        **kwargs: object,
    ) -> None:
        """Create a transformer encoder block."""
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if attn_type not in {"multihead", "additive", "scaled_dot", "sliding_window"}:
            raise ValueError(f"Unknown attention type {attn_type}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.attn_type = attn_type
        self.window_size = window_size
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.drop2 = tf.keras.layers.Dropout(dropout)
        distances = np.abs(np.arange(156)[:, None] - np.arange(156)[None, :])
        self.local_mask = tf.constant(
            distances <= window_size, dtype=tf.bool
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply local attention and the feed-forward residual path."""
        normalized = self.norm1(x)
        mask = self.local_mask if self.attn_type == "sliding_window" else None
        attended = self.attn(normalized, normalized, attention_mask=mask, training=training)
        x = x + self.drop1(attended, training=training)
        feedforward = self.ffn(self.norm2(x), training=training)
        return x + self.drop2(feedforward, training=training)

    def get_config(self) -> dict[str, object]:
        """Return serializable layer configuration."""
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "attn_type": self.attn_type,
            "window_size": self.window_size,
        }


def build_tst(
    num_classes: int = 5,
    d_model: int = 16,
    num_heads: int = 2,
    feedforward_dim: int = 32,
    num_blocks: int = 2,
    dropout: float = 0.0,
    input_norm: bool = False,
    ff_dim: int | None = None,
    num_layers: int | None = None,
    embedding: str = "conv_sinusoidal",
    attention_window: int = 7,
) -> tf.keras.Model:
    """Build the convolutional Time Series Transformer classifier."""
    inputs = tf.keras.Input(shape=(156, 7), dtype=tf.float32, name="window")
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    x = tf.keras.layers.Conv1D(64, 9, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(d_model, 3, padding="same", activation="relu")(x)
    if input_norm:
        x = tf.keras.layers.LayerNormalization(axis=-1)(x)
    x = PositionalEmbedding(
        sequence_length=156,
        d_model=d_model,
        method="sinusoidal" if embedding != "learned" else "embedding",
    )(x)
    for block_index in range(num_layers or num_blocks):
        x = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim or feedforward_dim,
            dropout=dropout,
            window_size=attention_window,
            name=f"encoder_{block_index}",
        )(x)
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
