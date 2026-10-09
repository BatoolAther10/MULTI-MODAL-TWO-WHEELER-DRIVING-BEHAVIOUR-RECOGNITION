"""Train a compact ResNet1D model and export it to ONNX."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windowing import make_windows


class BasicBlock1D(nn.Module):
    """Residual 1D block used in the compact ResNet style model."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        """Create a residual block with a 1D convolution and batch-normalization stack."""
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolutional mapping to the input sequence."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet1D(nn.Module):
    """Compact 1D CNN based on stacked residual blocks."""

    def __init__(self, num_classes: int = 5, channels: int = 16, blocks: int = 2) -> None:
        """Create the compact ResNet1D model."""
        super().__init__()
        self.input = nn.Conv1d(7, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.blocks = nn.Sequential(*(BasicBlock1D(channels) for _ in range(blocks)))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a sequence window to class logits."""
        x = x.transpose(1, 2)
        x = self.relu(self.bn(self.input(x)))
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def train_resnet1d(
    csv_path: str = "data/processed/clean.csv",
    model_path: str = "models/resnet1d.onnx",
    logs_path: str = "logs/resnet1d_metrics.json",
) -> dict:
    """Train the ResNet1D model and export it to ONNX."""
    data_path = ROOT / csv_path
    if not data_path.exists():
        raise FileNotFoundError(f"Clean CSV not found: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)

    label_order = sorted(np.unique(y).tolist())
    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_int,
        test_size=0.2,
        random_state=42,
        stratify=y_int,
    )

    model = ResNet1D(num_classes=len(label_order))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_train_t = torch.tensor(X_train.astype(np.float32))
    y_train_t = torch.tensor(y_train.astype(np.int64))
    X_test_t = torch.tensor(X_test.astype(np.float32))
    y_test_t = torch.tensor(y_test.astype(np.int64))

    model.train()
    for _ in range(15):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        preds = logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    output_path = ROOT / model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 156, 7, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "label_order": label_order,
        "model_path": str(output_path),
        "n_windows": int(X.shape[0]),
    }
    log_path = ROOT / logs_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"ResNet1D accuracy: {acc:.4f}")
    print(f"ResNet1D macro F1: {f1:.4f}")
    print(f"Saved ONNX model to: {output_path}")
    return metrics


if __name__ == "__main__":
    train_resnet1d()
