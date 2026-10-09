"""Run robustness experiments for the five strongest tabular models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_all import artifact_path, build_adapter, load_config
from utils.windowing import make_windows

TABULAR_MODELS = {"logistic_regression", "rbf_svm", "random_forest", "histgradientboosting", "lightgbm", "xgboost"}
FEATURE_NAMES = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Speed"]


def load_windows(config: dict[str, Any], limit: int = 500) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the first evaluation windows and sorted class labels."""
    with np.load(ROOT / config["data"]["windows_npz"], allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float32)[:limit]
        y = np.asarray(data["y"]).astype(str)[:limit]
    return X, y, sorted(np.unique(y).tolist())


def labels_to_int(y: np.ndarray, labels: list[str]) -> np.ndarray:
    """Encode string labels using one stable class order."""
    mapping = {label: index for index, label in enumerate(labels)}
    return np.asarray([mapping[label] for label in y], dtype=np.int64)


def top_models(config: dict[str, Any]) -> list[str]:
    """Read the five strongest classification models from the unified benchmark."""
    benchmark = pd.read_csv(ROOT / config["logging"]["benchmark_all_csv"])
    eligible = benchmark[benchmark["status"] == "benchmarked"].sort_values("macro_f1", ascending=False)
    models = [name for name in eligible["model"] if name in TABULAR_MODELS]
    if len(models) < 5:
        raise ValueError(f"Expected five tabular top models, found {models}.")
    return models[:5]


def build_estimator(name: str, seed: int) -> object:
    """Construct one tabular estimator for a window-size retraining run."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=100, class_weight="balanced", random_state=seed)
    if name == "rbf_svm":
        return SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=40, class_weight="balanced", n_jobs=-1, random_state=seed)
    if name == "histgradientboosting":
        return HistGradientBoostingClassifier(max_iter=50, random_state=seed)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=50, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.05, objective="multi:softprob", eval_metric="mlogloss", random_state=seed, n_jobs=1)
    raise ValueError(f"Unsupported robustness model: {name}")


def predictions_for_adapter(name: str, X: np.ndarray, y: np.ndarray, labels: list[str], config: dict[str, Any]) -> tuple[float, float]:
    """Run one saved adapter on perturbed windows."""
    artifact = artifact_path(name, ROOT / config["model"]["models_dir"])
    y_true = labels_to_int(y, labels)
    flat = X.reshape(len(X), -1)
    if name in TABULAR_MODELS:
        if name == "xgboost":
            from xgboost import XGBClassifier
            estimator = XGBClassifier()
            estimator.load_model(artifact)
            predictions = estimator.predict(flat).astype(np.int64)
        else:
            bundle = joblib.load(artifact)
            estimator = bundle["estimator"]
            if hasattr(estimator, "predict_proba"):
                predictions = np.argmax(estimator.predict_proba(flat), axis=1)
            else:
                predictions = estimator.predict(flat)
    else:
        adapter = build_adapter(name, artifact, len(labels), labels)
        predictions = np.asarray([int(np.argmax(adapter.predict(window))) for window in X], dtype=np.int64)
    return float(accuracy_score(y_true, predictions)), float(f1_score(y_true, predictions, average="macro", zero_division=0))


def train_and_score(name: str, X: np.ndarray, y: np.ndarray, labels: list[str], seed: int) -> tuple[float, float]:
    """Train one tabular model on regenerated windows and score it."""
    estimator = build_estimator(name, seed)
    encoded = labels_to_int(y, labels)
    flat = X.reshape(len(X), -1)
    estimator.fit(flat, encoded)
    if hasattr(estimator, "predict_proba"):
        predictions = np.argmax(estimator.predict_proba(flat), axis=1)
    else:
        predictions = estimator.predict(flat)
    return float(accuracy_score(encoded, predictions)), float(f1_score(encoded, predictions, average="macro", zero_division=0))


def save_plot(frame: pd.DataFrame, x_column: str, y_column: str, title: str, path: Path) -> None:
    """Save one robustness summary plot at 300 dpi."""
    plt.figure(figsize=(8, 5))
    for model_name, group in frame.groupby("model"):
        plt.plot(group[x_column], group[y_column], marker="o", label=model_name)
    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel(y_column.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def run_window_size(config: dict[str, Any], models: list[str], seed: int) -> pd.DataFrame:
    """Run the window-size retraining sweep."""
    source = pd.read_csv(ROOT / config["data"]["clean_csv"], encoding="utf-8-sig")
    rows: list[dict[str, object]] = []
    for window_size in config["robustness"]["window_sizes"]:
        stride = max(1, int(window_size) // 2)
        X, y, _ = make_windows(source, window_size=int(window_size), stride=stride)
        X, y = X[: int(config["robustness"]["evaluation_windows"])], y[: int(config["robustness"]["evaluation_windows"])]
        labels = sorted(np.unique(y).tolist())
        for model_name in models:
            accuracy, macro_f1 = train_and_score(model_name, X, y, labels, seed)
            rows.append({"model": model_name, "window_size": int(window_size), "accuracy": accuracy, "macro_f1": macro_f1})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "logs" / "robustness_window_size.csv", index=False)
    save_plot(frame, "window_size", "accuracy", "Robustness: Window Size", ROOT / "logs" / "figures" / "robustness_window_size.png")
    return frame


def run_perturbation(config: dict[str, Any], models: list[str], X: np.ndarray, y: np.ndarray, operation: str) -> pd.DataFrame:
    """Run one perturbation family against saved model artifacts."""
    seed = int(config["experiment"]["seed"])
    rows: list[dict[str, object]] = []
    if operation == "noise":
        levels = config["robustness"]["noise_snr_db"]
        for snr in levels:
            signal_power = float(np.mean(np.square(X[:, :, :6])))
            noise_power = signal_power / (10.0 ** (float(snr) / 10.0))
            rng = np.random.default_rng(seed + int(snr))
            perturbed = X.copy()
            perturbed[:, :, :6] += rng.normal(0.0, np.sqrt(noise_power), perturbed[:, :, :6].shape).astype(np.float32)
            for name in models:
                accuracy, macro_f1 = predictions_for_adapter(name, perturbed, y, sorted(np.unique(y).tolist()), config)
                rows.append({"model": name, "snr_db": snr, "accuracy": accuracy, "macro_f1": macro_f1})
        frame = pd.DataFrame(rows)
        filename, x_column, title = "robustness_noise.csv", "snr_db", "Robustness: Gaussian Noise"
        plotname = "robustness_noise.png"
    elif operation == "missing":
        levels = config["robustness"]["missing_rates"]
        for rate in levels:
            rng = np.random.default_rng(seed + int(float(rate) * 1000))
            perturbed = X.copy()
            mask = rng.random(perturbed.shape) < float(rate)
            perturbed[mask] = np.nan
            for window in perturbed:
                for channel in range(window.shape[1]):
                    series = pd.Series(window[:, channel]).ffill().bfill().fillna(0.0)
                    window[:, channel] = series.to_numpy(dtype=np.float32)
            for name in models:
                accuracy, macro_f1 = predictions_for_adapter(name, perturbed, y, sorted(np.unique(y).tolist()), config)
                rows.append({"model": name, "drop_rate": rate, "accuracy": accuracy, "macro_f1": macro_f1})
        frame = pd.DataFrame(rows)
        filename, x_column, title = "robustness_missing.csv", "drop_rate", "Robustness: Missing Data"
        plotname = "robustness_missing.png"
    elif operation == "channel_dropout":
        for channel_index, channel_name in enumerate(FEATURE_NAMES):
            perturbed = X.copy()
            perturbed[:, :, channel_index] = 0.0
            for name in models:
                accuracy, macro_f1 = predictions_for_adapter(name, perturbed, y, sorted(np.unique(y).tolist()), config)
                rows.append({"model": name, "channel": channel_name, "accuracy": accuracy, "macro_f1": macro_f1})
        frame = pd.DataFrame(rows)
        filename, x_column, title = "robustness_channel_dropout.csv", "channel", "Robustness: Channel Dropout"
        plotname = "robustness_channel_dropout.png"
    elif operation == "clock_jitter":
        for offset_ms in config["robustness"]["clock_offsets_ms"]:
            shift = max(1, int(round(float(offset_ms) * float(config["runtime"]["sensor_sample_rate_hz"]) / 1000.0)))
            perturbed = np.roll(X, shift=shift, axis=1)
            for name in models:
                accuracy, macro_f1 = predictions_for_adapter(name, perturbed, y, sorted(np.unique(y).tolist()), config)
                rows.append({"model": name, "offset_ms": offset_ms, "accuracy": accuracy, "macro_f1": macro_f1})
        frame = pd.DataFrame(rows)
        filename, x_column, title = "robustness_clock_jitter.csv", "offset_ms", "Robustness: Clock Jitter"
        plotname = "robustness_clock_jitter.png"
    else:
        raise ValueError(f"Unsupported perturbation: {operation}")
    frame.to_csv(ROOT / "logs" / filename, index=False)
    if x_column == "channel":
        pivot = frame.groupby("channel", as_index=False)["accuracy"].mean()
        plt.figure(figsize=(8, 5))
        plt.bar(pivot["channel"], pivot["accuracy"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Mean Accuracy")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(ROOT / "logs" / "figures" / plotname, dpi=300)
        plt.close()
    else:
        save_plot(frame, x_column, "accuracy", title, ROOT / "logs" / "figures" / plotname)
    return frame


def run_robustness() -> dict[str, pd.DataFrame]:
    """Run all five configured robustness experiments."""
    config = load_config()
    models = top_models(config)
    X, y, _ = load_windows(config, limit=int(config["robustness"].get("evaluation_windows", 500)))
    figures_dir = ROOT / config["logging"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["experiment"]["seed"])
    results = {"window_size": run_window_size(config, models, seed)}
    for operation in ("noise", "missing", "channel_dropout", "clock_jitter"):
        results[operation] = run_perturbation(config, models, X, y, operation)
    print("Top models:", ", ".join(models))
    for name, frame in results.items():
        print(f"{name}: {len(frame)} rows")
    return results


if __name__ == "__main__":
    run_robustness()
