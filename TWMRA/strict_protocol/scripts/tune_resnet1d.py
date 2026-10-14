"""Tune ResNet1D separately and plot architecture variation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_resnet1d import ResNet1D
from utils.windowing import make_windows

CONFIGS = [
    {"channels": 8, "blocks": 1},
    {"channels": 16, "blocks": 2},
    {"channels": 24, "blocks": 2},
]


def tune_resnet1d(epochs: int = 8) -> pd.DataFrame:
    """Train ResNet1D candidates and save metrics plus variation plots."""
    torch.manual_seed(42)
    df = pd.read_csv(ROOT / "data/processed/clean.csv", encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    labels = sorted(np.unique(y).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_int = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y_int, test_size=0.2, random_state=42, stratify=y_int
    )
    train_x = torch.tensor(X_train)
    train_y = torch.tensor(y_train)
    test_x = torch.tensor(X_test)
    rows = []

    for config in CONFIGS:
        torch.manual_seed(42)
        model = ResNet1D(num_classes=len(labels), **config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(train_x), train_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(test_x).argmax(dim=1).numpy()
        timings = []
        for window in X_test[:5]:
            start = time.perf_counter()
            with torch.no_grad():
                model(torch.tensor(window[None, ...]))
            timings.append((time.perf_counter() - start) * 1000.0)
        row = {
            **config,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "mae": float(np.mean(np.abs(y_test.astype(np.float32) - predictions.astype(np.float32)))),
            "latency_ms": float(np.median(timings)),
            "num_params": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        rows.append(row)
        print(row)

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    output = ROOT / "logs/resnet1d_tuning.csv"
    results.to_csv(output, index=False)
    (ROOT / "logs/resnet1d_tuning_best.json").write_text(
        json.dumps(results.iloc[0].to_dict(), indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_values = results["channels"].astype(str) + " / " + results["blocks"].astype(str)
    axes[0].plot(x_values, results["accuracy"], marker="o")
    axes[0].set_title("ResNet1D hyperparameters vs accuracy")
    axes[0].set_xlabel("channels / blocks")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x_values, results["mae"], marker="o", color="#c44e52")
    axes[1].set_title("ResNet1D hyperparameters vs MAE")
    axes[1].set_xlabel("channels / blocks")
    axes[1].set_ylabel("Encoded-label MAE")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure_path = ROOT / "logs/figures/tuning/resnet1d_tuning.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(results.to_string(index=False))
    print(f"Saved results to: {output}")
    print(f"Saved plot to: {figure_path}")
    return results


if __name__ == "__main__":
    tune_resnet1d()
