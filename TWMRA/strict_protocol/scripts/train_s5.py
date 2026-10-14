"""Train and export a compact S5-style state-space classifier."""

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
class S5Cell(tf.keras.layers.Layer):
    """Small diagonal state-space cell inspired by S5's linear recurrence."""

    def __init__(self, state_size: int = 16, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.state_size = state_size
        self.output_size = state_size

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        input_dim = int(input_shape[-1])
        self.log_decay = self.add_weight(name="log_decay", shape=(self.state_size,), initializer="zeros")
        self.input_projection = self.add_weight(
            name="input_projection", shape=(input_dim, self.state_size), initializer="glorot_uniform"
        )
        self.output_projection = self.add_weight(
            name="output_projection", shape=(self.state_size, self.state_size), initializer="glorot_uniform"
        )
        self.skip = self.add_weight(name="skip", shape=(input_dim, self.state_size), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, states: list[tf.Tensor]) -> tuple[tf.Tensor, list[tf.Tensor]]:
        state = states[0]
        decay = tf.sigmoid(self.log_decay)
        state = decay * state + tf.matmul(inputs, self.input_projection)
        output = tf.tanh(tf.matmul(state, self.output_projection) + tf.matmul(inputs, self.skip))
        return output, [state]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"state_size": self.state_size})
        return config


def build_s5(num_classes: int = 5, state_size: int = 16) -> tf.keras.Model:
    """Build a compact S5-style sequence classifier."""
    inputs = tf.keras.Input(shape=(156, 7), dtype=tf.float32, name="window")
    x = tf.keras.layers.Dense(state_size, name="input_projection")(inputs)
    x = tf.keras.layers.RNN(S5Cell(state_size), return_sequences=False, name="s5_state_space")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="s5")


def train_s5(
    csv_path: str = "data/processed/clean.csv",
    model_path: str = "models/s5.tflite",
    logs_path: str = "logs/s5_metrics.json",
) -> dict:
    """Train the S5-style model and export TFLite plus Keras fallback."""
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

    model = build_s5(num_classes=len(label_order))
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
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
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
    print(f"S5 parameters: {model.count_params()}")
    print(f"S5 accuracy: {accuracy:.4f}")
    print(f"S5 macro F1: {macro_f1:.4f}")
    print(f"Saved TFLite model to: {output_path}")
    return metrics


if __name__ == "__main__":
    train_s5()
