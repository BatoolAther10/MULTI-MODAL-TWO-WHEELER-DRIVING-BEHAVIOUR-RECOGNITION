"""Train and export compact S4D, S5, and LRU classifiers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windowing import make_windows


class DiagonalSSM(nn.Module):
    """Compact diagonal state-space recurrence used by S4D and S5 variants."""

    def __init__(self, input_dim: int, state_dim: int, mode: str) -> None:
        """Create a diagonal recurrent state-space layer."""
        super().__init__()
        self.mode = mode
        self.input_projection = nn.Linear(input_dim, state_dim)
        self.output_projection = nn.Linear(state_dim, state_dim)
        self.log_decay = nn.Parameter(torch.zeros(state_dim))
        self.skip = nn.Linear(input_dim, state_dim)
        if mode == "lru":
            self.phase = nn.Parameter(torch.zeros(state_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the selected state-space recurrence over time."""
        state = torch.zeros(inputs.shape[0], self.log_decay.shape[0], device=inputs.device)
        decay = torch.sigmoid(self.log_decay)
        outputs: list[torch.Tensor] = []
        for step in range(inputs.shape[1]):
            projected = self.input_projection(inputs[:, step])
            if self.mode == "lru":
                projected = projected * torch.cos(self.phase)
            state = decay * state + projected
            outputs.append(torch.tanh(self.output_projection(state) + self.skip(inputs[:, step])))
        return torch.stack(outputs, dim=1)


class SSMClassifier(nn.Module):
    """Sequence classifier built around a compact state-space recurrence."""

    def __init__(self, num_classes: int, mode: str, state_dim: int = 32) -> None:
        """Create an S4D, S5, or LRU classifier."""
        super().__init__()
        self.ssm = DiagonalSSM(7, state_dim, mode)
        self.norm = nn.LayerNorm(state_dim)
        self.head = nn.Linear(state_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map a sequence window to class logits."""
        sequence = self.norm(self.ssm(inputs))
        return self.head(sequence[:, -1])


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
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
    return X.astype(np.float32), y_int, labels


def train_one(name: str, mode: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, labels: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """Train and export one state-space classifier."""
    seed = int(config["experiment"]["seed"])
    torch.manual_seed(seed)
    model = SSMClassifier(len(labels), mode)
    features = torch.tensor(X_train)
    targets = torch.tensor(y_train)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test)).argmax(dim=1).numpy()
    models_dir = ROOT / config["model"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 156, 7),
        str(path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=18,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "label_order": labels,
        "model_path": str(path),
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    (ROOT / "logs" / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"{name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def train_ssm() -> dict[str, dict[str, Any]]:
    """Train S4D, S5, and LRU state-space variants."""
    config = load_config()
    X, y, labels = load_data(config)
    seed = int(config["experiment"]["seed"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    variants = {"s4d": "s4d", "s5_onnx": "s5", "lru": "lru"}
    return {name: train_one(name, mode, X_train, y_train, X_test, y_test, labels, config) for name, mode in variants.items()}


if __name__ == "__main__":
    train_ssm()
