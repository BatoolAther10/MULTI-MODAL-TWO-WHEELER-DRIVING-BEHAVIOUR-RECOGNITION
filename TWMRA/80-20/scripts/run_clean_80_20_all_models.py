"""Train and evaluate the 27-model clean.csv random 80/20 experiment."""

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.comprehensive_workflow import (
    ONNX_MODELS,
    TABULAR,
    _train_onnx,
    artifact_path,
    encode,
    estimator_for,
    load_config,
    predict_saved,
    tabular_matrix,
)
from scripts.neural_models_experiment import build_tf, tf_fit_final
from utils.windowing import make_windows

MODELS = [
    "histgradientboosting", "lightgbm", "random_forest", "xgboost",
    "logistic_regression", "linear_svm", "rbf_svm", "mlp_tabular",
    "conv_transformer", "gru", "lstm", "lstm_attention", "bilstm",
    "bilstm_attention", "tcn", "fusformer", "mega", "convnet_har",
    "tcnca_v1", "tcnca_v2", "tcnca_v3", "tst", "s4d", "s5_onnx",
    "lru", "sgconv", "resnet1d",
]

TABULAR_PARAMS: dict[str, dict[str, Any]] = {
    "histgradientboosting": {"max_iter": 240, "learning_rate": 0.025, "max_leaf_nodes": 15, "l2_regularization": 0.1},
    "lightgbm": {"n_estimators": 450, "learning_rate": 0.015, "num_leaves": 31, "min_child_samples": 10},
    "random_forest": {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2, "max_features": None},
    "xgboost": {"n_estimators": 320, "max_depth": 3, "learning_rate": 0.025, "min_child_weight": 1},
    "logistic_regression": {"C": 1.0},
    "linear_svm": {"C": 1.0},
    "rbf_svm": {"C": 4.0, "gamma": "scale"},
}

NEURAL_PARAMS: dict[str, dict[str, Any]] = {
    "conv_transformer": {"normalize": True},
    "gru": {"learning_rate": 0.0005, "weight_decay": 1e-5, "dropout": 0.1, "width": 96, "secondary_width": 48, "attention_heads": 4, "normalize": True},
    "lstm": {"learning_rate": 0.001, "weight_decay": 1e-4, "dropout": 0.2, "width": 32, "secondary_width": 16, "attention_heads": 2, "normalize": True},
    "lstm_attention": {"learning_rate": 0.001, "weight_decay": 1e-4, "dropout": 0.2, "width": 32, "secondary_width": 16, "attention_heads": 2, "normalize": True},
    "bilstm": {"learning_rate": 0.0005, "weight_decay": 1e-5, "dropout": 0.1, "width": 96, "secondary_width": 48, "attention_heads": 4, "normalize": True},
    "bilstm_attention": {"learning_rate": 0.0005, "weight_decay": 1e-5, "dropout": 0.1, "width": 96, "secondary_width": 48, "attention_heads": 4, "normalize": True},
    "tcn": {"learning_rate": 0.0005, "weight_decay": 1e-5, "dropout": 0.1, "channels": 48, "blocks": 5, "normalize": True},
    "fusformer": {"normalize": True},
    "mega": {"normalize": True},
    "convnet_har": {"normalize": True},
    "tcnca_v1": {"normalize": True},
    "tcnca_v2": {"normalize": True},
    "tcnca_v3": {"normalize": True},
    "tst": {"learning_rate": 0.001, "weight_decay": 1e-5, "d_model": 16, "num_heads": 2, "feedforward_dim": 64, "num_blocks": 1, "dropout": 0.1, "attention_window": 15, "normalize": True},
}


def save_confusion(path: Path, model: str, matrix: np.ndarray, classes: list[str]) -> None:
    """Append one confusion matrix to a long-format CSV."""
    rows = []
    for row_index, label in enumerate(classes):
        rows.append({"model": model, "true_class": label, "predictions": json.dumps(matrix[row_index].tolist())})
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run(epochs: int = 3, selected_models: list[str] | None = None) -> Path:
    """Run every configured model on one stratified clean.csv 80/20 split."""
    config = load_config()
    output = ROOT / "80/20"
    artifact_dir = output / "artifacts"
    figure_dir = output / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / config["data"]["clean_csv"], encoding="utf-8-sig")
    windows, labels_text, _ = make_windows(
        frame, int(config["runtime"]["window_size"]), int(config["runtime"]["stride"]),
        config["runtime"]["feature_columns"],
    )
    classes = list(config["runtime"]["classes"])
    labels = encode(labels_text.astype(str), classes)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    x_train, x_test = windows[train_idx].astype(np.float32), windows[test_idx].astype(np.float32)
    y_train, y_test = labels[train_idx], labels[test_idx]
    y_train_text = labels_text.astype(str)[train_idx]
    results: list[dict[str, Any]] = []
    classwise_path = output / "classwise.csv"
    confusion_path = output / "confusion_matrices.csv"
    models_to_run = selected_models or MODELS
    for model_name in models_to_run:
        print(f"Starting {model_name}", flush=True)
        started = time.perf_counter()
        try:
            params = dict(TABULAR_PARAMS.get(model_name, NEURAL_PARAMS.get(model_name, {"epochs": epochs})))
            if model_name in TABULAR:
                estimator = estimator_for(model_name, 42, len(classes), params)
                estimator.fit(tabular_matrix(x_train), y_train)
                predicted = np.asarray(estimator.predict(tabular_matrix(x_test)), dtype=np.int64)
                artifact = artifact_dir / f"{model_name}.joblib"
                joblib.dump({"estimator": estimator, "label_order": classes}, artifact)
            elif model_name == "mlp_tabular":
                estimator = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.01, learning_rate_init=0.001,
                                          max_iter=epochs, random_state=42)
                estimator.fit(x_train.reshape(len(x_train), -1), y_train)
                predicted = np.asarray(estimator.predict(x_test.reshape(len(x_test), -1)), dtype=np.int64)
                artifact = artifact_dir / f"{model_name}.joblib"
                joblib.dump({"estimator": estimator, "label_order": classes}, artifact)
            elif model_name in ONNX_MODELS:
                artifact = artifact_path(model_name, artifact_dir)
                _train_onnx(model_name, {"combined": (x_train, y_train_text)}, classes, artifact, 42, epochs)
                predicted, latency = predict_saved(model_name, artifact, x_test, classes)
            else:
                artifact = artifact_dir / f"{model_name}.keras"
                model, _ = tf_fit_final(model_name, (x_train, y_train), classes, params, artifact, 42, epochs)
                predicted = np.argmax(model.predict(x_test, verbose=0), axis=1).astype(np.int64)
            accuracy = float(accuracy_score(y_test, predicted))
            macro_f1 = float(f1_score(y_test, predicted, average="macro", zero_division=0))
            report = classification_report(y_test, predicted, labels=list(range(len(classes))),
                                           target_names=classes, output_dict=True, zero_division=0)
            for label in classes:
                classwise_row = report[label]
                pd.DataFrame([{"model": model_name, "class": label, **classwise_row}]).to_csv(
                    classwise_path, mode="a", header=not classwise_path.exists(), index=False)
            save_confusion(confusion_path, model_name, confusion_matrix(y_test, predicted, labels=list(range(len(classes)))), classes)
            elapsed = (time.perf_counter() - started) * 1000.0
            results.append({"model": model_name, "status": "completed", "accuracy": accuracy,
                            "macro_f1": macro_f1, "weighted_f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
                            "hyperparameters": json.dumps(params, sort_keys=True), "train_windows": len(x_train),
                            "test_windows": len(x_test), "fit_and_predict_ms": elapsed, "artifact": str(artifact.relative_to(ROOT))})
            print(f"Completed {model_name}: {accuracy:.4%}", flush=True)
        except Exception as exc:
            results.append({"model": model_name, "status": "failed", "accuracy": "FAILED",
                            "macro_f1": "FAILED", "weighted_f1": "FAILED",
                            "hyperparameters": json.dumps(TABULAR_PARAMS.get(model_name, NEURAL_PARAMS.get(model_name, {})), sort_keys=True),
                            "train_windows": len(x_train), "test_windows": len(x_test),
                            "fit_and_predict_ms": (time.perf_counter() - started) * 1000.0,
                            "artifact": "", "error": repr(exc)})
            print(f"FAILED {model_name}: {exc}", flush=True)
    results_frame = pd.DataFrame(results)
    results_path = output / "results.csv"
    if selected_models and results_path.exists():
        prior = pd.read_csv(results_path)
        prior = prior[~prior["model"].isin(selected_models)]
        results_frame = pd.concat([prior, results_frame], ignore_index=True)
        results_frame = results_frame.set_index("model").reindex(MODELS).reset_index()
    results_frame.to_csv(results_path, index=False)
    (output / "protocol.json").write_text(json.dumps({
        "source": "data/processed/clean.csv", "split": "stratified random 80/20",
        "random_state": 42, "window_size": 156, "stride": 78,
        "feature_columns": config["runtime"]["feature_columns"], "classes": classes,
        "train_windows": len(x_train), "test_windows": len(x_test),
    }, indent=2), encoding="utf-8")
    print(results_frame[["model", "status", "accuracy", "macro_f1"]].to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    arguments = parser.parse_args()
    run(arguments.epochs, arguments.models)
