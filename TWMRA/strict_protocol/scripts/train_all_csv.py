"""Train feasible model families using the four root CSV split protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windowing import make_windows
from inference_layer.model_factory import create_model

TABULAR_MODELS = {
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "histgradientboosting",
    "lightgbm",
    "xgboost",
    "mlp_tabular",
}
NEURAL_MODELS = {"gru", "lstm"}
ANOMALY_MODELS = {"dtaad"}
NON_TRAINABLE_MODELS = {"stub", "egodrive_rt", "egodrive_max"}


def load_config() -> dict[str, Any]:
    """Load the project YAML configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(config: dict[str, Any], key: str, default: str) -> Path:
    """Resolve a configured path relative to the project root."""
    value = config["data"].get(key, default)
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_split(path: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read one BOM-safe CSV and independently generate its windows."""
    if not path.exists():
        raise FileNotFoundError(f"Required split CSV not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    runtime = config["runtime"]
    return make_windows(
        frame,
        window_size=int(runtime["window_size"]),
        stride=int(runtime["stride"]),
        feature_cols=list(runtime["feature_columns"]),
    )


def encode_labels(
    y_train: np.ndarray, y_other: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Encode labels with a fixed class order and reject unknown classes."""
    mapping = {label: index for index, label in enumerate(labels)}
    unknown = sorted(set(map(str, y_other)) - set(mapping))
    if unknown:
        raise ValueError(f"Labels absent from training split: {unknown}")
    return (
        np.asarray([mapping[str(label)] for label in y_train], dtype=np.int64),
        np.asarray([mapping[str(label)] for label in y_other], dtype=np.int64),
    )


def build_tabular(name: str, seed: int, class_count: int) -> Any:
    """Build one supported tabular estimator."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC

    if name == "logistic_regression":
        return LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced", random_state=seed, max_iter=1000)
    if name == "rbf_svm":
        return SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    if name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass", num_class=class_count, n_estimators=200,
            learning_rate=0.05, class_weight="balanced", random_state=seed,
            n_jobs=-1, verbosity=-1,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob", num_class=class_count, n_estimators=200,
            max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, eval_metric="mlogloss", random_state=seed, n_jobs=1,
        )
    raise ValueError(f"No tabular builder for {name}")


def predict_estimator(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Return integer predictions from a sklearn-compatible estimator."""
    return np.asarray(estimator.predict(features), dtype=np.int64)


def metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate accuracy and macro-F1 for one split."""
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
    }


def artifact_path(name: str, models_dir: Path) -> Path:
    """Return the deployment artifact path for a model."""
    return models_dir / ("xgboost_tabular.json" if name == "xgboost" else f"{name}.pkl")


def save_estimator(name: str, estimator: Any, labels: list[str], path: Path) -> None:
    """Persist an estimator in the format used by the existing adapters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "xgboost":
        estimator.save_model(path)
        path.with_name("xgboost_tabular_labels.json").write_text(
            json.dumps(labels), encoding="utf-8"
        )
    else:
        joblib.dump({"estimator": estimator, "label_order": labels}, path)


def evaluate_saved(
    name: str, path: Path, windows: np.ndarray, y_true: np.ndarray, labels: list[str]
) -> dict[str, float]:
    """Load a deployment artifact through its project adapter and score it."""
    kwargs: dict[str, Any] = {}
    if name == "xgboost":
        kwargs = {"num_classes": len(labels), "label_order": labels}
    adapter = create_model(name, **kwargs)
    adapter.load(str(path))
    predictions = np.asarray(
        [int(np.argmax(adapter.predict(window))) for window in windows], dtype=np.int64
    )
    return metrics(y_true, predictions)


def run(
    config: dict[str, Any], requested: list[str] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute validation selection, combined retraining, and test evaluation."""
    data = {
        key: resolve_path(config, key, filename)
        for key, filename in {
            "root_train_csv": "train.csv",
            "root_val_csv": "val.csv",
            "root_train_val_csv": "train_val_combined.csv",
            "root_test_csv": "test.csv",
        }.items()
    }
    windows = {key: load_split(path, config) for key, path in data.items()}
    labels = sorted(set(map(str, windows["root_train_csv"][1])))
    for key in ("root_val_csv", "root_train_val_csv", "root_test_csv"):
        labels = sorted(set(labels) | set(map(str, windows[key][1])))
    train_y, _ = encode_labels(windows["root_train_csv"][1], windows["root_train_csv"][1], labels)
    val_y = encode_labels(windows["root_train_csv"][1], windows["root_val_csv"][1], labels)[1]
    combined_y = encode_labels(windows["root_train_csv"][1], windows["root_train_val_csv"][1], labels)[1]
    test_y = encode_labels(windows["root_train_csv"][1], windows["root_test_csv"][1], labels)[1]
    names = requested or list(config.get("benchmark", {}).get("models", []))
    seed = int(config.get("experiment", {}).get("seed", 42))
    models_dir = ROOT / config["model"].get("models_dir", "models")
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "protocol": "independent_windows_from_root_csvs",
        "files": {key: str(value) for key, value in data.items()},
        "feature_columns": list(config["runtime"]["feature_columns"]),
        "window_size": int(config["runtime"]["window_size"]),
        "stride": int(config["runtime"]["stride"]),
        "labels": labels,
        "window_counts": {key: int(value[0].shape[0]) for key, value in windows.items()},
        "seed": seed,
        "limitations": [
            "Each registered family is evaluated with its default builder configuration; no hyperparameter sweep is defined.",
            "Families without a root-CSV builder are reported explicitly and are not replaced by pre-existing artifacts.",
            "Window labels use majority vote independently within each CSV split.",
        ],
    }
    counts = {
        "n_train_windows": int(len(train_y)),
        "n_val_windows": int(len(val_y)),
        "n_combined_windows": int(len(combined_y)),
        "n_test_windows": int(len(test_y)),
    }
    for name in names:
        base = {
            "model": name, "configuration": "default", "selected_by_val": False,
            "val_macro_f1": np.nan, "test_macro_f1": np.nan, **counts,
        }
        if name in ANOMALY_MODELS or name in NON_TRAINABLE_MODELS:
            rows.append({**base, "status": "anomaly_only" if name in ANOMALY_MODELS else "non_trainable",
                         "message": "No supervised classification builder for this model family."})
            continue
        if name not in TABULAR_MODELS:
            rows.append({**base, "status": "not_trainable", "message": "No root-CSV training builder implemented."})
            continue
        try:
            x_train = windows["root_train_csv"][0].reshape(len(train_y), -1)
            x_val = windows["root_val_csv"][0].reshape(len(val_y), -1)
            estimator = build_tabular(name, seed, len(labels))
            estimator.fit(x_train, train_y)
            val_metrics = metrics(val_y, predict_estimator(estimator, x_val))
            base.update({"status": "validated", "val_macro_f1": val_metrics["macro_f1"],
                         "val_accuracy": val_metrics["accuracy"], "selected_by_val": True})
            combined_x = windows["root_train_val_csv"][0].reshape(len(combined_y), -1)
            estimator.fit(combined_x, combined_y)
            path = artifact_path(name, models_dir)
            save_estimator(name, estimator, labels, path)
            test_metrics = evaluate_saved(name, path, windows["root_test_csv"][0], test_y, labels)
            base.update({"status": "evaluated", "test_macro_f1": test_metrics["macro_f1"],
                         "test_accuracy": test_metrics["accuracy"], "artifact": str(path)})
        except Exception as exc:  # Keep a visible row for optional dependencies/failures.
            base.update({"status": "not_trainable", "message": f"{type(exc).__name__}: {exc}"})
        rows.append(base)
    result = pd.DataFrame(rows)
    output = ROOT / config["logging"].get("root_split_results_csv", "logs/root_split_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metadata["result_csv"] = str(output)
    metadata["selected_models"] = result.loc[result["status"] == "evaluated"].sort_values(
        "val_macro_f1", ascending=False
    )["model"].tolist()
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return result, metadata


def main() -> None:
    """Parse command-line arguments and run the root split experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Restrict execution to named configured models.")
    args = parser.parse_args()
    result, _ = run(load_config(), args.models)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
