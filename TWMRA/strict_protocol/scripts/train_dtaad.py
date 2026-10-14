"""Train and evaluate a dual TCN-attention autoencoder for novelty scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windowing import make_windows


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_windows(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load window features and string labels from the clean dataset."""
    df = pd.read_csv(ROOT / config["data"]["clean_csv"], encoding="utf-8-sig")
    X, y, _ = make_windows(
        df,
        window_size=int(config["runtime"]["window_size"]),
        stride=int(config["runtime"]["stride"]),
    )
    return X.astype(np.float32), y.astype(str)


def dilated_branch(inputs: tf.Tensor, name: str) -> tf.Tensor:
    """Build one dilated temporal convolution branch."""
    x = tf.keras.layers.Conv1D(16, 3, padding="same", dilation_rate=1, activation="relu", name=f"{name}_d1")(inputs)
    x = tf.keras.layers.Conv1D(16, 3, padding="same", dilation_rate=2, activation="relu", name=f"{name}_d2")(x)
    return tf.keras.layers.Conv1D(16, 3, padding="same", dilation_rate=4, activation="relu", name=f"{name}_d4")(x)


def build_dtaad() -> tf.keras.Model:
    """Build the dual TCN-attention reconstruction autoencoder."""
    inputs = tf.keras.Input(shape=(156, 7), dtype=tf.float32, name="window")
    branch_a = dilated_branch(inputs, "branch_a")
    branch_b = dilated_branch(inputs, "branch_b")
    fused = tf.keras.layers.Concatenate(name="dual_branch_fusion")([branch_a, branch_b])
    attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16, name="temporal_attention")(fused, fused)
    encoded = tf.keras.layers.Add(name="attention_residual")([fused, attention])
    decoded = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu", name="decoder_conv")(encoded)
    outputs = tf.keras.layers.Conv1D(7, 1, padding="same", name="reconstruction")(decoded)
    return tf.keras.Model(inputs, outputs, name="dtaad")


def reconstruction_errors(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Calculate one mean squared reconstruction error per window."""
    reconstructed = model.predict(X, batch_size=64, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=(1, 2)).astype(np.float32)


def export_tflite(model: tf.keras.Model, path: Path) -> None:
    """Export the autoencoder as float16 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    path.write_bytes(converter.convert())


def train_dtaad() -> dict[str, Any]:
    """Train DTAAD on STRAIGHT windows and save novelty metrics."""
    config = load_config()
    seed = int(config["experiment"]["seed"])
    tf.keras.utils.set_random_seed(seed)
    X, labels = load_windows(config)
    normal_label = str(config.get("anomaly", {}).get("normal_label", "STRAIGHT"))
    normal_mask = labels == normal_label
    if normal_mask.sum() < 2:
        raise ValueError(f"Need at least two normal windows labelled {normal_label}.")
    normal_X = X[normal_mask]
    model = build_dtaad()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(normal_X, normal_X, validation_split=0.2, epochs=15, batch_size=32, verbose=0)

    normal_errors = reconstruction_errors(model, normal_X)
    all_errors = reconstruction_errors(model, X)
    threshold = float(np.quantile(normal_errors, 0.95))
    anomaly_targets = (labels != normal_label).astype(np.int32)
    auc = float(roc_auc_score(anomaly_targets, all_errors)) if len(np.unique(anomaly_targets)) == 2 else float("nan")
    models_dir = ROOT / config["model"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    keras_path = models_dir / "dtaad.keras"
    tflite_path = models_dir / "dtaad.tflite"
    model.save(keras_path)
    export_tflite(model, tflite_path)
    metrics = {
        "model": "dtaad",
        "normal_label": normal_label,
        "normal_train_windows": int(normal_X.shape[0]),
        "threshold_p95": threshold,
        "mean_normal_error": float(np.mean(normal_errors)),
        "mean_anomaly_error": float(np.mean(all_errors[~normal_mask])) if (~normal_mask).any() else float("nan"),
        "roc_auc": auc,
        "params": int(model.count_params()),
        "model_path": str(tflite_path),
        "fallback_path": str(keras_path),
    }
    (ROOT / "logs" / "dtaad_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(ROOT / "logs" / "benchmark_dtaad.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_dtaad()
