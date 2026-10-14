"""Train and save all Group A tabular baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_training_data(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the random training split and encode labels."""
    split_path = ROOT / config["data"]["splits_npz"]
    with np.load(split_path, allow_pickle=False) as data:
        X = np.asarray(data["random_X_train"], dtype=np.float32).reshape(len(data["random_X_train"]), -1)
        y = np.asarray(data["random_y_train"]).astype(str)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return X, y_encoded, encoder.classes_.tolist()


def build_sklearn_models(seed: int) -> dict[str, object]:
    """Construct the sklearn-compatible baseline estimators."""
    return {
        "logistic_regression": LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=seed, max_iter=1000),
        "rbf_svm": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "histgradientboosting": HistGradientBoostingClassifier(random_state=seed),
    }


def save_bundle(name: str, estimator: object, labels: list[str], models_dir: Path) -> Path:
    """Save a sklearn estimator and its class order as a joblib bundle."""
    path = models_dir / f"{name}.pkl"
    joblib.dump({"estimator": estimator, "label_order": labels}, path)
    return path


def train_lightgbm(X: np.ndarray, y: np.ndarray, seed: int) -> object:
    """Train a balanced LightGBM multiclass classifier."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("Install lightgbm to train the LightGBM baseline.") from exc
    estimator = LGBMClassifier(
        objective="multiclass",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    estimator.fit(X, y)
    return estimator


def train_xgboost(X: np.ndarray, y: np.ndarray, class_count: int, seed: int) -> object:
    """Train the XGBoost multiclass classifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost to train the XGBoost baseline.") from exc
    estimator = XGBClassifier(
        objective="multi:softprob",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        num_class=class_count,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=1,
    )
    estimator.fit(X, y)
    return estimator


def train_mlp_tabular(X: np.ndarray, y: np.ndarray, class_count: int, seed: int, output_path: Path) -> None:
    """Train and save a two-hidden-layer Keras tabular classifier."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install tensorflow to train MLP_tabular.") from exc
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(class_count, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2, verbose=0)
    model.save(output_path)


def train_all() -> list[Path]:
    """Train every configured Group A model and return artifact paths."""
    config = load_config()
    X, y, labels = load_training_data(config)
    seed = int(config.get("experiment", {}).get("seed", 42))
    models_dir = ROOT / config["model"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for name, estimator in build_sklearn_models(seed).items():
        print(f"Training {name}...", flush=True)
        estimator.fit(X, y)
        paths.append(save_bundle(name, estimator, labels, models_dir))

    print("Training lightgbm...", flush=True)
    lightgbm_model = train_lightgbm(X, y, seed)
    paths.append(save_bundle("lightgbm", lightgbm_model, labels, models_dir))

    print("Training xgboost...", flush=True)
    xgb_model = train_xgboost(X, y, len(labels), seed)
    xgb_path = models_dir / "xgboost_tabular.json"
    xgb_model.save_model(xgb_path)
    with (models_dir / "xgboost_tabular_labels.json").open("w", encoding="utf-8") as handle:
        json.dump(labels, handle)
    paths.append(xgb_path)

    mlp_path = models_dir / "mlp_tabular.keras"
    print("Training mlp_tabular...", flush=True)
    train_mlp_tabular(X, y, len(labels), seed, mlp_path)
    paths.append(mlp_path)
    for path in paths:
        print(f"Saved {path.name}")
    return paths


if __name__ == "__main__":
    train_all()
