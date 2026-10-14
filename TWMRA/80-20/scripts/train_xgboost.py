"""Train a window-level XGBoost classifier on the generated window data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windowing import make_windows


def train_xgboost(csv_path: str = "data/processed/clean.csv", model_path: str = "models/xgboost.json") -> dict:
    """Load windows, train the XGBoost model, and save the classifier to disk."""
    data_path = Path(csv_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Clean CSV not found: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    X, y, _ = make_windows(df, window_size=156, stride=78)
    X_flat = X.reshape(X.shape[0], -1)

    label_order = sorted(np.unique(y).tolist())
    y_int = pd.Series(y).astype(pd.CategoricalDtype(categories=label_order)).cat.codes.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y_int,
        test_size=0.2,
        random_state=42,
        stratify=y_int,
    )

    clf = XGBClassifier(
        objective="multi:softmax",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        num_class=len(label_order),
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(out_path)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "label_order": label_order,
        "n_windows": int(X_flat.shape[0]),
    }
    with open("logs/xgboost_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"XGBoost accuracy: {acc:.4f}")
    print(f"XGBoost macro F1: {f1:.4f}")
    print(f"Saved model to: {out_path}")
    return metrics


if __name__ == "__main__":
    train_xgboost()
