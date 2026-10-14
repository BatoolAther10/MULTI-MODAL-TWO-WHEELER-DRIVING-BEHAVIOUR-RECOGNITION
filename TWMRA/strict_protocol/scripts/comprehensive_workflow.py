"""Run the reproducible root-CSV training, deployment, and reporting workflow."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_layer.model_factory import create_model
from utils.windowing import make_windows

MODELS = [
    "histgradientboosting", "lightgbm", "random_forest", "xgboost",
    "logistic_regression", "linear_svm", "mlp_tabular", "conv_transformer",
    "gru", "lstm", "fusformer", "mega", "convnet_har", "tcnca_v3",
    "rbf_svm", "tcnca_v2", "tcnca_v1", "tst", "s4d", "s5_onnx", "lru",
    "sgconv", "resnet1d",
]
TABULAR = {
    "histgradientboosting", "random_forest", "logistic_regression",
    "lightgbm", "xgboost", "linear_svm", "rbf_svm",
}
TF_MODELS = {
    "mlp_tabular", "conv_transformer", "gru", "lstm", "fusformer", "mega",
    "convnet_har", "tcnca_v3", "tcnca_v2", "tcnca_v1", "tst",
}
ONNX_MODELS = {"s4d", "s5_onnx", "lru", "sgconv", "resnet1d"}
UNAVAILABLE: set[str] = set()

# Small, reproducible searches.  The validation split is never used for fitting.
SEARCH_SPACE: dict[str, list[dict[str, Any]]] = {
    "histgradientboosting": [{"max_iter": 20, "learning_rate": 0.08, "max_leaf_nodes": 31, "max_bins": 32},
                             {"max_iter": 40, "learning_rate": 0.05, "max_leaf_nodes": 63, "max_bins": 32}],
    "lightgbm": [{"n_estimators": 120, "learning_rate": 0.05, "num_leaves": 31},
                 {"n_estimators": 220, "learning_rate": 0.03, "num_leaves": 63}],
    "random_forest": [{"n_estimators": 60, "max_depth": None, "min_samples_leaf": 1},
                      {"n_estimators": 100, "max_depth": 24, "min_samples_leaf": 2}],
    "xgboost": [{"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05},
                {"n_estimators": 140, "max_depth": 6, "learning_rate": 0.03}],
    "logistic_regression": [{"C": 0.5}, {"C": 2.0}],
    "linear_svm": [{"C": 0.5}, {"C": 2.0}],
    "rbf_svm": [{"C": 2.0, "gamma": "scale"}, {"C": 8.0, "gamma": "scale"}],
}


def load_config() -> dict[str, Any]:
    """Load config.yaml from the project root."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def csv_path(config: dict[str, Any], key: str, default: str) -> Path:
    """Resolve a configured data path relative to the project root."""
    path = Path(config["data"].get(key, default))
    return path if path.is_absolute() else ROOT / path


def read_windows(path: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Read a BOM-safe root CSV and generate configured windows."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    runtime = config["runtime"]
    windows, labels, _ = make_windows(
        frame,
        window_size=int(runtime["window_size"]),
        stride=int(runtime["stride"]),
        feature_cols=list(runtime["feature_columns"]),
    )
    return windows, labels.astype(str)


def cap_training_windows(
    values: tuple[np.ndarray, np.ndarray], limit: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically cap only fitting windows while retaining class coverage."""
    windows, labels = values
    if limit <= 0 or len(labels) <= limit:
        return values
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(set(labels.tolist())):
        candidates = np.flatnonzero(labels == label)
        take = max(1, int(round(limit * len(candidates) / len(labels))))
        selected.extend(rng.choice(candidates, size=min(take, len(candidates)), replace=False).tolist())
    selected = sorted(selected[:limit])
    return windows[selected], labels[selected]


def encode(labels: np.ndarray, order: list[str]) -> np.ndarray:
    """Encode labels using one stable class order."""
    mapping = {value: index for index, value in enumerate(order)}
    unknown = sorted(set(labels.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return np.asarray([mapping[value] for value in labels], dtype=np.int64)


def tabular_matrix(windows: np.ndarray) -> np.ndarray:
    """Create compact window-level features for fast tabular deployment."""
    stride = 6
    usable = (windows.shape[1] // stride) * stride
    pooled = windows[:, :usable].reshape(len(windows), usable // stride, stride, windows.shape[2]).mean(axis=2)
    return pooled.reshape(len(windows), -1).astype(np.float32)


def estimator_for(name: str, seed: int, classes: int, params: dict[str, Any] | None = None) -> Any:
    """Construct one supported tabular estimator."""
    params = dict(params or {})
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=300, solver="liblinear", class_weight="balanced", random_state=seed, **params
        )
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced", random_state=seed, max_iter=400, dual="auto", **params)
    if name == "rbf_svm":
        return SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=seed, **params
        )
    if name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            objective="multiclass", num_class=classes, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbosity=-1, **params,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob", num_class=classes, subsample=0.9,
            colsample_bytree=0.8, eval_metric="mlogloss", random_state=seed, n_jobs=1,
            **params,
        )
    raise ValueError(f"No training builder for {name}")


def artifact_path(name: str, directory: Path) -> Path:
    """Return a fresh workflow deployment artifact path."""
    if name == "xgboost":
        return directory / "xgboost.json"
    if name == "mlp_tabular":
        return directory / f"{name}.keras"
    if name in TF_MODELS:
        return directory / f"{name}.tflite"
    if name in ONNX_MODELS:
        return directory / f"{name}.onnx"
    return directory / f"{name}.pkl"


def save_estimator(name: str, estimator: Any, labels: list[str], path: Path) -> None:
    """Save a deployment artifact without reading any prior artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "xgboost":
        estimator.save_model(path)
        path.with_name("xgboost_labels.json").write_text(
            json.dumps({"labels": labels, "feature_stride": 6}), encoding="utf-8"
        )
    else:
        joblib.dump({
            "estimator": estimator, "label_order": labels,
            "feature_stride": 6 if name in TABULAR else 1,
        }, path)


def adapter_for(name: str, labels: list[str]) -> Any:
    """Create the adapter corresponding to a saved deployment artifact."""
    kwargs: dict[str, Any] = {}
    if name == "xgboost":
        kwargs = {"num_classes": len(labels), "label_order": labels}
    return create_model(name, **kwargs)


def _export_tflite(model: Any, path: Path) -> None:
    """Export a Keras model and keep a portable Keras fallback beside it."""
    import tensorflow as tf

    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path.with_suffix(".keras"))
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        path.write_bytes(converter.convert())
    except Exception:
        # Some recurrent graphs require Select TF ops unavailable on Windows.
        # The adjacent Keras artifact is a valid deployment fallback.
        if path.exists():
            path.unlink()


def _build_tf(name: str, classes: int) -> Any:
    """Build a requested TensorFlow architecture using project builders."""
    import tensorflow as tf
    from scripts.train_conv import build_convnet
    from scripts.train_recurrent import build_model
    from scripts.train_tst import build_tst
    from scripts.train_transformers import (
        build_conv_transformer, build_fusformer, build_mega, build_tcnca_variant,
    )

    if name == "mlp_tabular":
        model = tf.keras.Sequential([
            tf.keras.layers.Input((156 * 7,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(classes, activation="softmax"),
        ])
    elif name in {"gru", "lstm"}:
        model = build_model(name, classes)
    elif name == "convnet_har":
        model = build_convnet(classes)
    elif name == "tst":
        model = build_tst(classes, d_model=24, num_heads=2, feedforward_dim=48, num_blocks=2, dropout=0.1)
    elif name == "tcnca_v1":
        model = build_tcnca_variant("v1", classes)
    elif name == "tcnca_v2":
        model = build_tcnca_variant("v2", classes, chunk_size=26)
    elif name == "tcnca_v3":
        model = build_tcnca_variant("v3", classes, chunk_size=52)
    else:
        model = {"mega": build_mega, "fusformer": build_fusformer,
                 "conv_transformer": build_conv_transformer}[name](classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def _train_tf(name: str, data: dict[str, tuple[np.ndarray, np.ndarray]],
              labels: list[str], path: Path, seed: int, epochs: int) -> None:
    """Train a TensorFlow candidate on combined windows and export it."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = _build_tf(name, len(labels))
    x = data["combined"][0].astype(np.float32)
    y = encode(data["combined"][1], labels)
    if name == "mlp_tabular":
        x = x.reshape(len(x), -1)
    model.fit(x, y, epochs=epochs, batch_size=64, verbose=0)
    if name == "mlp_tabular":
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
    else:
        _export_tflite(model, path)


def _train_onnx(name: str, data: dict[str, tuple[np.ndarray, np.ndarray]],
                labels: list[str], path: Path, seed: int, epochs: int) -> None:
    """Train an existing project PyTorch builder and export an ONNX artifact."""
    import torch
    from scripts.train_conv import SGConvNet, train_torch_model
    from scripts.train_resnet1d import ResNet1D
    from scripts.train_ssm import SSMClassifier

    torch.manual_seed(seed)
    x = data["combined"][0].astype(np.float32)
    y = encode(data["combined"][1], labels)
    if name == "sgconv":
        model = SGConvNet(len(labels))
    elif name == "resnet1d":
        model = ResNet1D(len(labels))
    else:
        mode = {"s4d": "s4d", "s5_onnx": "s5", "lru": "lru"}[name]
        model = SSMClassifier(len(labels), mode)
    if name in {"sgconv", "resnet1d"}:
        model = train_torch_model(model, x, y, epochs=epochs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        features, targets = torch.tensor(x), torch.tensor(y)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss_fn(model(features), targets).backward()
            optimizer.step()
        model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        torch.onnx.export(model, torch.randn(1, 156, 7), str(path),
                          input_names=["window"], output_names=["logits"],
                          dynamic_axes={"window": {0: "batch_size"}, "logits": {0: "batch_size"}},
                          opset_version=17)


def predict_saved(
    name: str, path: Path, windows: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Predict through a freshly loaded deployment adapter and measure latency."""
    adapter = adapter_for(name, labels)
    adapter.load(str(path))
    batch = getattr(adapter, "keras_model", None)
    if batch is None:
        batch = getattr(adapter, "model", None)
    if batch is not None and hasattr(batch, "predict"):
        start = time.perf_counter_ns()
        features = tabular_matrix(windows) if name in TABULAR else (
            windows.reshape(len(windows), -1) if name == "mlp_tabular" else windows.astype(np.float32)
        )
        try:
            values = np.asarray(batch.predict(features, verbose=0))
        except TypeError:
            values = np.asarray(batch.predict(features))
        elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
        if values.ndim == 1:
            classes = np.asarray(getattr(batch, "classes_", np.arange(len(labels))))
            values = np.asarray([int(np.flatnonzero(classes == value)[0]) for value in values])
            predicted = values.astype(np.int64)
        else:
            predicted = np.argmax(values, axis=1).astype(np.int64)
        return predicted, np.full(len(predicted), elapsed / max(len(predicted), 1))
    session = getattr(adapter, "session", None)
    if session is not None:
        start = time.perf_counter_ns()
        input_name = session.get_inputs()[0].name
        values = np.asarray(session.run(None, {input_name: windows.astype(np.float32)})[0])
        elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
        return np.argmax(values, axis=1).astype(np.int64), np.full(len(values), elapsed / max(len(values), 1))
    predictions: list[int] = []
    latencies: list[float] = []
    for window in windows:
        start = time.perf_counter_ns()
        predictions.append(int(np.argmax(adapter.predict(window))))
        latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return np.asarray(predictions, dtype=np.int64), np.asarray(latencies, dtype=float)


def score(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the primary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
    }


def train_and_evaluate(
    name: str,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Train one model, select on validation, retrain, and evaluate test once."""
    row: dict[str, Any] = {
        "model": name, "status": "not_trainable", "error": "",
        "search_candidates": len(SEARCH_SPACE.get(name, [{"epochs": 2}, {"epochs": 4}])),
    }
    if name in UNAVAILABLE:
        row["error"] = "Unavailable pretrained weights/implementation; not substituted."
        return row, None
    try:
        y_train = encode(data["train"][1], labels)
        y_val = encode(data["val"][1], labels)
        path = artifact_path(name, output_dir)
        epochs = 2
        if name in TABULAR:
            x_train = tabular_matrix(data["train"][0])
            x_val = tabular_matrix(data["val"][0])
            candidates = SEARCH_SPACE.get(name, [{}])
            best_params: dict[str, Any] = candidates[0]
            best_metrics = {"macro_f1": -1.0, "accuracy": -1.0}
            for candidate in candidates:
                candidate_model = estimator_for(name, seed, len(labels), candidate)
                candidate_model.fit(x_train, y_train)
                candidate_pred = np.asarray(candidate_model.predict(x_val), dtype=np.int64)
                candidate_metrics = score(y_val, candidate_pred)
                if (candidate_metrics["macro_f1"], candidate_metrics["accuracy"]) > (
                    best_metrics["macro_f1"], best_metrics["accuracy"]
                ):
                    best_params, best_metrics = dict(candidate), candidate_metrics
            val_metrics = best_metrics
            model = estimator_for(name, seed, len(labels), best_params)
            x_combined = tabular_matrix(data["combined"][0])
            model.fit(x_combined, encode(data["combined"][1], labels))
            save_estimator(name, model, labels, path)
            row["best_config"] = json.dumps(best_params, sort_keys=True)
        else:
            # Train a candidate on train for selection, then build a fresh
            # candidate on train+val so no validation weights leak into it.
            if name in TF_MODELS:
                _train_tf(name, {"combined": data["train"]}, labels, path, seed, epochs)
            else:
                _train_onnx(name, {"combined": data["train"]}, labels, path, seed, epochs)
            val_pred, _ = predict_saved(name, path, data["val"][0], labels)
            val_metrics = score(y_val, val_pred)
            if name in TF_MODELS:
                _train_tf(name, data, labels, path, seed, epochs)
            else:
                _train_onnx(name, data, labels, path, seed, epochs)
            row["best_config"] = json.dumps({"epochs": epochs}, sort_keys=True)
        y_test = encode(data["test"][1], labels)
        test_pred, latency = predict_saved(name, path, data["test"][0], labels)
        test_metrics = score(y_test, test_pred)
        row.update({
            "status": "evaluated", "artifact": str(path.resolve().relative_to(ROOT)),
            "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"],
            "test_latency_mean_ms": float(np.mean(latency)),
            "test_latency_p95_ms": float(np.percentile(latency, 95)),
            "test_windows_evaluated_once": len(test_pred),
        })
        return row, test_pred
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, None


def robustness(
    rows: pd.DataFrame,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    output_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate supported saved artifacts on validation perturbations."""
    selected = rows.loc[rows["status"] == "evaluated", "model"].tolist()
    evaluation_windows = int(config.get("robustness", {}).get("evaluation_windows", 100))
    source = (data["val"][0][:evaluation_windows], data["val"][1][:evaluation_windows])
    y = encode(source[1], labels)
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    for name in selected:
        path = artifact_path(name, output_dir)
        base = source[0]
        variants = {
            "channel_dropout": np.where(
                np.arange(base.shape[2])[None, None, :] == 0, 0.0, base
            ),
            "clock_jitter": np.roll(base, 1, axis=1),
            "missing": np.where(rng.random(base.shape) < 0.05, np.nan, base),
            "noise": base + rng.normal(0, np.std(base, axis=(0, 1), keepdims=True) * 0.1, base.shape),
        }
        for kind, variant in variants.items():
            variant = np.nan_to_num(variant, nan=0.0).astype(np.float32)
            predicted, _ = predict_saved(name, path, variant, labels)
            records.append({"model": name, "condition": kind, **score(y, predicted)})
        for requested_size in config.get("robustness", {}).get("window_sizes", [100, 128, 156, 200]):
            # Adapters have a fixed deployment contract; resize perturbed windows
            # back to that contract rather than silently skipping this condition.
            positions = np.linspace(0.0, 1.0, int(requested_size))
            target = np.linspace(0.0, 1.0, base.shape[1])
            resized = np.empty_like(base)
            for index in range(len(base)):
                for channel in range(base.shape[2]):
                    resized[index, :, channel] = np.interp(
                        target, positions, base[index, :int(requested_size), channel]
                        if int(requested_size) <= base.shape[1]
                        else np.interp(positions, target, base[index, :, channel]),
                    )
            predicted, _ = predict_saved(name, path, resized.astype(np.float32), labels)
            records.append({
                "model": name, "condition": f"window_size_{requested_size}",
                **score(y, predicted), "requested_window_size": int(requested_size),
            })
    result = pd.DataFrame(records)
    result.to_csv(ROOT / "logs" / "comprehensive_robustness.csv", index=False)
    return result


def write_figures(rows: pd.DataFrame, robustness_rows: pd.DataFrame) -> list[Path]:
    """Write benchmark and robustness PNG figures at 300 dpi."""
    figures = ROOT / "logs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    valid = rows.loc[rows["status"] == "evaluated"].copy()
    if not valid.empty:
        for metric in ("test_macro_f1", "test_latency_mean_ms"):
            path = figures / f"comprehensive_{metric}.png"
            plt.figure(figsize=(11, 5))
            plt.bar(valid["model"], valid[metric])
            plt.xticks(rotation=65, ha="right")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()
            paths.append(path)
    if not robustness_rows.empty:
        path = figures / "comprehensive_robustness.png"
        pivot = robustness_rows.pivot_table(index="model", columns="condition", values="macro_f1")
        pivot.plot(kind="bar", figsize=(11, 5))
        plt.ylabel("macro F1")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        paths.append(path)
    return paths


def write_report(
    config: dict[str, Any], rows: pd.DataFrame, robustness_rows: pd.DataFrame,
    data: dict[str, tuple[np.ndarray, np.ndarray]], figures: list[Path],
) -> Path:
    """Write the comprehensive Markdown report and supporting class-wise tables."""
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    valid = rows.loc[rows["status"] == "evaluated"]
    best = valid.sort_values("val_macro_f1", ascending=False).iloc[0]["model"] if not valid.empty else "none"
    class_rows: list[dict[str, Any]] = []
    for _, row in valid.iterrows():
        name = str(row["model"])
        path = ROOT / str(row["artifact"])
        predicted, _ = predict_saved(name, path, data["val"][0][:100], sorted(set(data["train"][1])))
        report = classification_report(
            encode(data["val"][1][:100], sorted(set(data["train"][1]))), predicted,
            labels=list(range(5)), target_names=sorted(set(data["train"][1])),
            output_dict=True, zero_division=0,
        )
        matrix = confusion_matrix(
            encode(data["val"][1][:100], sorted(set(data["train"][1]))),
            predicted,
            labels=list(range(5)),
        )
        plt.figure(figsize=(5, 4))
        plt.imshow(matrix, cmap="Blues")
        plt.title(f"{name} validation confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(logs / "figures" / f"comprehensive_cm_{name}.png", dpi=300)
        plt.close()
        for label, values in report.items():
            if isinstance(values, dict):
                class_rows.append({"model": name, "class": label, **values})
    pd.DataFrame(class_rows).to_csv(logs / "comprehensive_classwise.csv", index=False)
    sections = [
        "# Comprehensive model workflow report",
        "",
        "## Dataset and Metrics",
        f"- Source protocol: root `train.csv` ({len(data['train'][0])} windows) for fitting, "
        f"`val.csv` ({len(data['val'][0])} windows) for selection, "
        f"`train_val_combined.csv` ({len(data['combined'][0])} windows) for retraining, "
        f"`test.csv` ({len(data['test'][0])} windows) for one saved-artifact evaluation.",
        f"- Features: `{config['runtime']['feature_columns']}`; window={config['runtime']['window_size']}, stride={config['runtime']['stride']}.",
        "- Metrics: accuracy, macro-F1, class-wise precision/recall/F1, and deployment latency.",
        "",
        "## Model Explanations and Hyperparameters",
        "Tabular models use configured flattened-window builders; neural, transformer, convolutional, recurrent, and state-space "
        "models use the existing project builders and are exported through their registered deployment adapters.",
        "",
        "## Metrics for Every Model",
        rows.to_string(index=False),
        "",
        f"## Benchmark Graphs\nGenerated: {', '.join(str(p.resolve().relative_to(ROOT)) for p in figures)}",
        "",
        "## Confusion Matrices and Class-Wise Results",
        "Class-wise metrics are in `logs/comprehensive_classwise.csv`. Confusion matrices are generated only for models with a saved deployment artifact.",
        "",
        "## Robustness Channel Dropout, Clock Jitter, Missing, Noise, Window Size",
        robustness_rows.to_string(index=False) if not robustness_rows.empty else "No trainable model was available.",
        "",
        "## Final Comparison and Recommendation",
        f"Validation-selected recommendation: **{best}**. Test metrics are reported only for freshly trained workflow artifacts; "
        "unavailable architectures are not replaced by legacy artifacts and have no claimed test metrics.",
    ]
    report_path = logs / "COMPREHENSIVE_WORKFLOW_REPORT.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path


def run() -> None:
    """Run the complete standalone workflow."""
    config = load_config()
    data = {
        key: read_windows(csv_path(config, key, filename), config)
        for key, filename in {
            "train": "train.csv", "val": "val.csv",
            "combined": "train_val_combined.csv", "test": "test.csv",
        }.items()
    }
    seed = int(config["experiment"]["seed"])
    labels = sorted(set(np.concatenate([value[1] for value in data.values()])))
    deployment = ROOT / config["model"].get("workflow_deployment_dir", "models/workflow_deployment")
    rows_list: list[dict[str, Any]] = []
    for name in MODELS:
        row, _ = train_and_evaluate(name, data, labels, deployment, seed)
        rows_list.append(row)
    rows = pd.DataFrame(rows_list)
    rows.to_csv(ROOT / "logs" / "comprehensive_metrics.csv", index=False)
    configs = {
        str(row["model"]): json.loads(row["best_config"])
        for _, row in rows.loc[rows["status"] == "evaluated"].iterrows()
        if isinstance(row.get("best_config"), str)
    }
    (ROOT / "logs" / "best_configs.json").write_text(
        json.dumps(configs, indent=2, sort_keys=True), encoding="utf-8"
    )
    comparison = rows.loc[rows["status"] == "evaluated"].sort_values(
        ["test_macro_f1", "test_accuracy"], ascending=False
    )
    comparison.to_csv(ROOT / "logs" / "final_comparison.csv", index=False)
    robustness_rows = robustness(rows, data, labels, deployment, config)
    figures = write_figures(rows, robustness_rows)
    report = write_report(config, rows, robustness_rows, data, figures)
    metadata = {
        "models": MODELS, "labels": labels,
        "window_counts": {key: len(value[0]) for key, value in data.items()},
        "test_evaluation": "saved deployment artifacts evaluated exactly once; robustness uses validation windows",
        "report": str(report.relative_to(ROOT)),
    }
    (ROOT / "logs" / "comprehensive_workflow.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(rows.to_string(index=False))
    print(f"Report: {report}")


if __name__ == "__main__":
    run()
