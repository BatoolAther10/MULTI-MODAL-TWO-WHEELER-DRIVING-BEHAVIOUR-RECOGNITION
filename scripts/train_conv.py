"""Train and export the ResNet1D, ConvNet-HAR, and SGConv models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_resnet1d import ResNet1D
from utils.windowing import make_windows


class SGConvNet(nn.Module):
    """Compact structured state-space convolution approximation for export."""

    def __init__(self, num_classes: int = 5) -> None:
        """Create a depthwise dilated convolution classifier."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=5, padding=2, groups=1),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=9, padding=8, dilation=2, groups=32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, padding=16, dilation=4, groups=32),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a sequence window to class logits."""
        features = self.features(x.transpose(1, 2))
        return self.head(self.pool(features).squeeze(-1))


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_data(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load windowed features and integer labels."""
    df = pd.read_csv(ROOT / config["data"]["clean_csv"], encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    labels = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    return X.astype(np.float32), np.asarray([label_to_index[label] for label in y], dtype=np.int64), labels


def train_torch_model(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> nn.Module:
    """Train a compact PyTorch convolutional classifier."""
    model.train()
    features = torch.tensor(X)
    targets = torch.tensor(y)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()
    return model.eval()


def save_torch_onnx(model: nn.Module, path: Path) -> None:
    """Export a PyTorch sequence model to ONNX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 156, 7),
        str(path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )


def build_convnet(class_count: int) -> tf.keras.Model:
    """Build the three-block ConvNet-HAR classifier."""
    inputs = tf.keras.Input(shape=(156, 7))
    hidden = tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu")(inputs)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(hidden)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(hidden)
    hidden = tf.keras.layers.GlobalAveragePooling1D()(hidden)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(hidden)
    model = tf.keras.Model(inputs, outputs, name="convnet_har")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_tflite(model: tf.keras.Model, path: Path) -> None:
    """Export a Keras model as float16 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    path.write_bytes(converter.convert())


def train_conv() -> dict[str, dict[str, Any]]:
    """Train all three convolutional model variants and save metrics."""
    config = load_config()
    seed = int(config["experiment"]["seed"])
    tf.keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    X, y, labels = load_data(config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    models_dir = ROOT / config["model"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    for name, model in (("resnet1d", ResNet1D(len(labels))), ("sgconv", SGConvNet(len(labels)))):
        trained = train_torch_model(model, X_train, y_train)
        with torch.no_grad():
            predictions = trained(torch.tensor(X_test)).argmax(dim=1).numpy()
        path = models_dir / f"{name}.onnx"
        save_torch_onnx(trained, path)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "label_order": labels,
            "model_path": str(path),
            "params": int(sum(parameter.numel() for parameter in trained.parameters())),
        }

    convnet = build_convnet(len(labels))
    convnet.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=0)
    predictions = np.argmax(convnet.predict(X_test, verbose=0), axis=1)
    keras_path = models_dir / "convnet_har.keras"
    tflite_path = models_dir / "convnet_har.tflite"
    convnet.save(keras_path)
    export_tflite(convnet, tflite_path)
    results["convnet_har"] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "label_order": labels,
        "model_path": str(tflite_path),
        "params": int(convnet.count_params()),
    }
    for name, metrics in results.items():
        (ROOT / "logs" / f"{name}_conv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"{name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    return results


if __name__ == "__main__":
    train_conv()
