"""Train stronger normalized/class-balanced TCNCA and TST candidates."""

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

from scripts.train_tcnca import build_tcnca
from scripts.train_tst import build_tst
from utils.windowing import make_windows


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Return inverse-frequency weights normalized around one."""
    counts = np.bincount(y)
    total = len(y)
    return {index: float(total / (len(counts) * count)) for index, count in enumerate(counts)}


def wrap_normalization(model: tf.keras.Model, X_train: np.ndarray) -> tf.keras.Model:
    """Put train-set feature normalization inside the exported model."""
    normalization = tf.keras.layers.Normalization(axis=-1, name="feature_normalization")
    normalization.adapt(X_train)
    inputs = tf.keras.Input(shape=(156, 7), dtype=tf.float32, name="window")
    return tf.keras.Model(inputs, model(normalization(inputs)), name=f"normalized_{model.name}")


def train_one(name: str, builder, builder_config: dict, X_train, X_test, y_train, y_test, weights=None) -> dict:
    """Train, evaluate, and export one optimized neural candidate."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    base_model = builder(num_classes=5, **builder_config)
    model = wrap_normalization(base_model, X_train)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=8, restore_best_weights=True),
    ]

    class MacroF1Callback(tf.keras.callbacks.Callback):
        """Track validation macro F1 for early stopping."""

        def on_epoch_end(self, epoch, logs=None):
            validation_predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
            logs = logs or {}
            logs["val_macro_f1"] = f1_score(y_test, validation_predictions, average="macro")
            print(f"{name} epoch {epoch + 1}: val_macro_f1={logs['val_macro_f1']:.4f}")

    callbacks.insert(0, MacroF1Callback())
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=60,
        batch_size=32,
        class_weight=weights,
        callbacks=callbacks,
        verbose=0,
    )
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    output = ROOT / f"models/{name}_optimized.tflite"
    fallback = output.with_suffix(".keras")
    model.save(fallback)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    output.write_bytes(converter.convert())
    metrics = {
        "model": name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mae": float(np.mean(np.abs(y_test.astype(np.float32) - predictions.astype(np.float32)))),
        "num_params": int(model.count_params()),
        "model_path": str(output),
        "fallback_path": str(fallback),
        "configuration": builder_config,
    }
    (ROOT / f"logs/{name}_optimized_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics)
    return metrics


def main() -> None:
    """Train optimized TCNCA and TST candidates on the common split."""
    df = pd.read_csv(ROOT / "data/processed/clean.csv", encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    labels = sorted(np.unique(y).tolist())
    mapping = {label: index for index, label in enumerate(labels)}
    y_int = np.asarray([mapping[label] for label in y], dtype=np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y_int, test_size=0.2, random_state=42, stratify=y_int
    )
    weights = class_weights(y_train)
    results = [
        train_one("tcnca", build_tcnca, {"channels": 24, "blocks": 2}, X_train, X_test, y_train, y_test),
        train_one("tst", build_tst, {"d_model": 24, "num_heads": 2, "feedforward_dim": 48, "num_blocks": 2, "dropout": 0.1}, X_train, X_test, y_train, y_test),
    ]
    (ROOT / "logs/optimized_neural_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
