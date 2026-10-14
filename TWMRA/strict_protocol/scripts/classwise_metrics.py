"""Generate classwise reports and confusion matrices for strong models."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_all import artifact_path, build_adapter, load_config


def load_benchmark_data(config: dict[str, Any], window_count: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the same full-window benchmark source used by benchmark_all."""
    with np.load(ROOT / config["data"]["windows_npz"], allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float32)[:window_count]
        y = np.asarray(data["y"]).astype(str)[:window_count]
    labels = sorted(np.unique(y).tolist())
    return X, y, labels


def predict_labels(adapter: object, X: np.ndarray, labels: list[str]) -> np.ndarray:
    """Run one loaded adapter and return integer predictions."""
    return np.asarray([int(np.argmax(adapter.predict(window))) for window in X], dtype=np.int64)


def save_classwise_table(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_dir: Path) -> pd.DataFrame:
    """Save precision, recall, F1, and support for one model."""
    report = classification_report(y_true, y_pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0)
    rows: list[dict[str, object]] = []
    for label in labels:
        values = report[label]
        rows.append({"model": model_name, "class": label, "precision": values["precision"], "recall": values["recall"], "f1": values["f1-score"], "support": int(values["support"])})
    for average_name in ("macro avg", "weighted avg"):
        values = report[average_name]
        rows.append({"model": model_name, "class": average_name, "precision": values["precision"], "recall": values["recall"], "f1": values["f1-score"], "support": int(values["support"])})
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / f"classwise_{model_name}.csv", index=False)
    return result


def save_confusion_plot(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], figures_dir: Path) -> None:
    """Save a 5x5 normalized and count-annotated confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"cm_{model_name}.png", dpi=300)
    plt.close()


def generate_classwise_reports() -> pd.DataFrame:
    """Generate all eligible classwise CSVs and confusion matrix figures."""
    config = load_config()
    benchmark = pd.read_csv(ROOT / config["logging"]["benchmark_all_csv"])
    eligible = benchmark[(benchmark["status"] == "benchmarked") & (benchmark["macro_f1"] > 0.75)]["model"].tolist()
    X, y, labels = load_benchmark_data(config, int(benchmark["window_count"].dropna().iloc[0]))
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
    models_dir = ROOT / config["model"]["models_dir"]
    figures_dir = ROOT / config["logging"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_tables: list[pd.DataFrame] = []
    for model_name in eligible:
        artifact = artifact_path(model_name, models_dir)
        if artifact is None or not artifact.exists():
            continue
        adapter = build_adapter(model_name, artifact, len(labels), labels)
        y_pred = predict_labels(adapter, X, labels)
        all_tables.append(save_classwise_table(model_name, y_true, y_pred, labels, ROOT / "logs"))
        save_confusion_plot(model_name, y_true, y_pred, labels, figures_dir)
    combined = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    combined.to_csv(ROOT / "logs" / "classwise_all.csv", index=False)
    class_rows = combined[combined["class"].isin(labels)]
    hardest = class_rows.groupby("class")["f1"].mean().sort_values().index[0] if not class_rows.empty else "unknown"
    print(f"Eligible models (macro F1 > 0.75): {', '.join(eligible)}")
    print(f"Hardest class by mean F1 across eligible models: {hardest}")
    print(f"Saved {len(all_tables)} classwise tables and confusion matrices.")
    return combined


if __name__ == "__main__":
    generate_classwise_reports()
