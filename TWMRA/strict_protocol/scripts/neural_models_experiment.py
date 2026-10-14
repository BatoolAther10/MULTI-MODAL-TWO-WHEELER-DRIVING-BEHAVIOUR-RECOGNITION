"""Leakage-safe experiment runner for all requested neural model families."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_layer.model_factory import create_model
from scripts.train_conv import build_convnet
from scripts.train_recurrent import build_model as build_recurrent
from scripts.train_tcnca import build_tcnca
from scripts.train_resnet1d import ResNet1D
from scripts.train_ssm import SSMClassifier
from scripts.train_transformers import (build_conv_transformer, build_fusformer,
                                        build_mega, build_tcnca_variant)
from scripts.train_tst import build_tst
from utils.windowing import make_windows


ARCHITECTURES = (
    "conv_transformer", "gru", "lstm", "lstm_attention", "bilstm", "bilstm_attention",
    "tcn", "fusformer", "mega", "convnet_har",
    "tcnca_v3", "tcnca_v2", "tcnca_v1", "tst", "s4d", "s5_onnx", "lru",
    "sgconv", "resnet1d",
)


def load_config() -> dict[str, Any]:
    """Load YAML configuration."""
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def load_split(path: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Read one split into the identical raw seven-channel model input."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    x, y, _ = make_windows(frame, int(config["runtime"]["window_size"]),
                            int(config["runtime"]["stride"]),
                            config["runtime"]["feature_columns"])
    return x.astype(np.float32), y.astype(str)


def cap_windows(
    x: np.ndarray, y: np.ndarray, limit: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically cap a split while retaining every available class."""
    if limit is None or len(y) <= limit:
        return x, y
    indices = np.arange(len(y))
    chosen, _ = train_test_split(
        indices, train_size=limit, random_state=seed, stratify=y
    )
    chosen.sort()
    return x[chosen], y[chosen]


def encode(y: np.ndarray, classes: list[str]) -> np.ndarray:
    """Encode labels using the fixed runtime class order."""
    mapping = {label: index for index, label in enumerate(classes)}
    return np.asarray([mapping[str(value)] for value in y], dtype=np.int64)


def build_tf(name: str, classes: int, params: dict[str, Any]) -> tf.keras.Model:
    """Build a project TensorFlow implementation for one architecture."""
    if name in ("gru", "lstm", "lstm_attention", "bilstm", "bilstm_attention"):
        return build_recurrent(name, classes, params)
    if name == "tcn":
        return build_tcnca(
            classes,
            channels=int(params.get("channels", 32)),
            blocks=int(params.get("blocks", 4)),
        )
    if name == "convnet_har":
        return build_convnet(classes)
    if name == "tst":
        model_keys = (
            "d_model", "num_heads", "feedforward_dim", "num_blocks", "dropout",
            "input_norm", "ff_dim", "num_layers", "embedding", "attention_window"
        )
        return build_tst(
            classes, **{key: params[key] for key in model_keys if key in params}
        )
    if name == "tcnca_v1":
        return build_tcnca_variant("v1", classes, dilations=params.get("dilations"))
    if name == "tcnca_v2":
        return build_tcnca_variant("v2", classes, chunk_size=26, dilations=params.get("dilations"))
    if name == "tcnca_v3":
        return build_tcnca_variant("v3", classes, chunk_size=52, dilations=params.get("dilations"))
    return {
        "conv_transformer": build_conv_transformer,
        "fusformer": build_fusformer,
        "mega": build_mega,
    }[name](classes)


def add_train_normalization(
    model: tf.keras.Model, x_train: np.ndarray
) -> tf.keras.Model:
    """Add a normalization layer fitted only on the current training split."""
    normalization = tf.keras.layers.Normalization(axis=-1, name="feature_normalization")
    normalization.adapt(x_train)
    inputs = tf.keras.Input(shape=(156, 7), dtype=tf.float32, name="window")
    return tf.keras.Model(inputs, model(normalization(inputs)), name=f"normalized_{model.name}")


def class_weights(y: np.ndarray, count: int) -> dict[int, float]:
    """Calculate balanced weights without peeking at validation or test labels."""
    values = np.unique(y)
    weights = compute_class_weight("balanced", classes=values, y=y)
    return {int(label): float(weight) for label, weight in zip(values, weights) if int(label) < count}


def tf_fit(name: str, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray],
           classes: list[str], params: dict[str, Any], out: Path, seed: int,
           epochs: int) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Tune and fit one TensorFlow model using validation macro-F1 checkpoints."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_tf(name, len(classes), params)
    if bool(params.get("normalize", False)):
        model = add_train_normalization(model, train[0])
    lr = float(params.get("learning_rate", 1e-3))
    model.compile(optimizer=tf.keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=float(params.get("weight_decay", 0.0))),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    xtr, ytr = train
    xv, yv = val
    val_f1 = MacroF1Callback(xv, yv)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(out), monitor="val_accuracy",
                                                     mode="max", save_best_only=True,
                                                     save_weights_only=False, verbose=0)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=5, restore_best_weights=False
    )
    model.fit(xtr, ytr, validation_data=val, epochs=epochs, batch_size=32,
              class_weight=class_weights(ytr, len(classes)),
              callbacks=[val_f1, checkpoint, early_stop],
              verbose=0)
    selected = tf.keras.models.load_model(out)
    return selected, {"best_val_macro_f1": float(val_f1.best), "best_val_accuracy": float(val_f1.best_accuracy), "epochs": len(val_f1.values)}


def tf_fit_final(
    name: str,
    train: tuple[np.ndarray, np.ndarray],
    classes: list[str],
    params: dict[str, Any],
    out: Path,
    seed: int,
    epochs: int,
) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Fit the selected neural configuration on every combined window."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_tf(name, len(classes), params)
    if bool(params.get("normalize", False)):
        model = add_train_normalization(model, train[0])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=float(params.get("learning_rate", 1e-3)),
            weight_decay=float(params.get("weight_decay", 0.0)),
        ),
        loss="sparse_categorical_crossentropy",
    )
    x_train, y_train = train
    model.fit(
        x_train,
        y_train,
        epochs=max(1, int(epochs)),
        batch_size=32,
        class_weight=class_weights(y_train, len(classes)),
        verbose=0,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return model, {"epochs": max(1, int(epochs)), "fit_windows": len(y_train)}


class MacroF1Callback(tf.keras.callbacks.Callback):
    """Compute validation macro-F1 after each epoch for checkpoint selection."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Store the frozen validation set."""
        super().__init__()
        self.x, self.y, self.best, self.best_accuracy, self.values = x, y, -1.0, -1.0, []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Record macro-F1 and stop after a patience window without improvement."""
        pred = np.argmax(self.model.predict(self.x, verbose=0), axis=1)
        value = float(f1_score(self.y, pred, average="macro", zero_division=0))
        if logs is not None:
            logs["val_macro_f1"] = value
        self.values.append(value)
        self.best = max(self.best, value)
        self.best_accuracy = max(self.best_accuracy, float((pred == self.y).mean()))
        if len(self.values) > 6 and self.best > max(self.values[-6:-1]):
            self.model.stop_training = True


def torch_model(name: str, classes: int, params: dict[str, Any]) -> Any:
    """Build a project PyTorch implementation for one architecture."""
    if name in ("s4d", "s5_onnx", "lru"):
        return SSMClassifier(classes, {"s4d": "s4d", "s5_onnx": "s5", "lru": "lru"}[name],
                             int(params.get("state_dim", 32)))
    if name == "resnet1d":
        return ResNet1D(classes, channels=int(params.get("width", 16)),
                        blocks=int(params.get("blocks", 2)))
    from scripts.train_conv import SGConvNet
    return SGConvNet(classes)


def metric(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Return accuracy, macro-F1, confusion matrix, and classwise metrics."""
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classwise": [{"class": label, "precision": float(cm[i, i] / max(cm[:, i].sum(), 1)),
                       "recall": float(cm[i, i] / max(cm[i].sum(), 1)),
                       "support": int(cm[i].sum())} for i, label in enumerate(classes)],
    }


def export_tf(model: tf.keras.Model, path: Path) -> Path:
    """Save Keras and best-effort float16 TFLite deployment artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keras = path.with_suffix(".keras")
    model.save(keras)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        path.write_bytes(converter.convert())
        return path
    except Exception:
        return keras


def batch_predictions(name: str, model: Any, windows: np.ndarray) -> np.ndarray:
    """Predict a complete split in batches without per-window framework overhead."""
    if name not in ("s4d", "s5_onnx", "lru", "sgconv", "resnet1d"):
        return np.argmax(model.predict(windows, batch_size=64, verbose=0), axis=1)
    import torch
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(windows, dtype=torch.float32))
    return torch.argmax(output, dim=1).cpu().numpy()


def run_torch(name: str, train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray],
              test: tuple[np.ndarray, np.ndarray], classes: list[str], params: dict[str, Any],
              path: Path, seed: int, epochs: int) -> tuple[Any, dict[str, Any]]:
    """Train a PyTorch architecture with weighted loss and validation early stopping."""
    import torch
    torch.manual_seed(seed)
    model = torch_model(name, len(classes), params)
    opt = torch.optim.AdamW(model.parameters(), lr=float(params.get("learning_rate", 1e-3)),
                            weight_decay=float(params.get("weight_decay", 1e-4)))
    weights = torch.tensor([class_weights(train[1], len(classes)).get(i, 1.0)
                           for i in range(len(classes))], dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    xtr, ytr = map(torch.tensor, train)
    best, best_score, stale = None, -1.0, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(xtr), ytr).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            score = float(f1_score(val[1], model(torch.tensor(val[0])).argmax(1).numpy(),
                                   average="macro", zero_division=0))
        if score > best_score:
            best_score, stale, best = score, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
    if best is not None:
        model.load_state_dict(best)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    try:
        torch.onnx.export(model, torch.randn(1, 156, 7), str(path), input_names=["window"],
                          output_names=["logits"], dynamic_axes={"window": {0: "batch_size"},
                          "logits": {0: "batch_size"}}, opset_version=17)
        artifact = path
    except Exception as exc:
        # A state dict is not a deployment artifact for an ONNX adapter.  Do
        # not create a misleading .pt file that a registered adapter might
        # later try to interpret as ONNX.
        raise RuntimeError(f"{name} ONNX export failed: {exc}") from exc
    return model, {"best_val_macro_f1": best_score, "artifact": str(artifact)}


def main() -> None:
    """Run tuning, frozen retraining, deployment evaluation, robustness, and DOCX reporting."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--max-trials", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    config = load_config(); seed = int(config["experiment"]["seed"])
    np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    out = ROOT / "logs" / "neural_models_experiment"; out.mkdir(parents=True, exist_ok=True)
    artifact_dir = ROOT / "models" / "neural_experiment"; artifact_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {k: ROOT / config["data"][k] for k in
                  ("root_train_csv", "root_val_csv", "root_train_val_csv", "root_test_csv")}
    raw = {key: load_split(path, config) for key, path in data_paths.items()}
    classes = list(config["runtime"]["classes"])
    train_limit = int(config["experiment"].get("max_training_windows_per_split", 0)) or None
    data = {}
    for key, (x, y) in raw.items():
        if key in ("root_train_csv", "root_val_csv"):
            x, y = cap_windows(x, y, train_limit, seed)
        data[key] = (x, encode(y, classes))
    grids = [
        {"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.2},
        {"learning_rate": 3e-4, "weight_decay": 1e-5, "dropout": 0.3},
        {"learning_rate": 5e-4, "weight_decay": 1e-5, "dropout": 0.1},
        {"learning_rate": 1e-4, "weight_decay": 1e-6, "dropout": 0.2},
    ]
    rows: list[dict[str, Any]] = []; selected: dict[str, Any] = {}
    for name in args.architectures:
        model_grids = grids
        if name in ("gru", "lstm", "lstm_attention", "bilstm", "bilstm_attention"):
            model_grids = [
                {**grids[0], "width": 32, "secondary_width": 16, "attention_heads": 2, "normalize": True},
                {**grids[1], "width": 64, "secondary_width": 32, "attention_heads": 2, "normalize": True},
                {**grids[2], "width": 96, "secondary_width": 48, "attention_heads": 4, "normalize": True},
                {**grids[3], "width": 128, "secondary_width": 64, "attention_heads": 4, "normalize": True},
            ]
        elif name == "tcn":
            model_grids = [
                {**grids[0], "channels": 16, "blocks": 3},
                {**grids[1], "channels": 32, "blocks": 4},
                {**grids[2], "channels": 48, "blocks": 5},
                {**grids[3], "channels": 64, "blocks": 6},
            ]
        elif name == "tst":
            model_grids = [
                {"learning_rate": 1e-3, "weight_decay": 1e-4, "d_model": 16, "num_heads": 2, "feedforward_dim": 64, "ff_dim": 64, "num_blocks": 1, "num_layers": 1, "dropout": .1, "input_norm": False, "attention_window": 7, "normalize": True},
                {"learning_rate": 5e-4, "weight_decay": 1e-5, "d_model": 24, "num_heads": 4, "feedforward_dim": 128, "ff_dim": 128, "num_blocks": 2, "num_layers": 2, "dropout": .2, "input_norm": True, "attention_window": 7, "normalize": True},
                {"learning_rate": 3e-4, "weight_decay": 1e-5, "d_model": 32, "num_heads": 4, "feedforward_dim": 256, "ff_dim": 256, "num_blocks": 4, "num_layers": 4, "dropout": .3, "input_norm": True, "attention_window": 7, "normalize": True},
                {"learning_rate": 1e-3, "weight_decay": 1e-5, "d_model": 16, "num_heads": 2, "feedforward_dim": 64, "ff_dim": 64, "num_blocks": 1, "num_layers": 1, "dropout": .1, "input_norm": False, "attention_window": 15, "normalize": True},
                {"learning_rate": 5e-4, "weight_decay": 1e-6, "d_model": 24, "num_heads": 2, "feedforward_dim": 96, "ff_dim": 96, "num_blocks": 2, "num_layers": 2, "dropout": .1, "input_norm": True, "attention_window": 15, "normalize": True},
            ]
        elif name == "resnet1d":
            model_grids = [{**grids[0], "width": 16, "blocks": 2},
                           {**grids[1], "width": 32, "blocks": 3}]
        elif name in ("s4d", "s5_onnx", "lru"):
            model_grids = [{**grids[0], "state_dim": 32},
                           {**grids[1], "state_dim": 64}]
        trials: list[dict[str, Any]] = []
        for index, params in enumerate(model_grids[:args.max_trials]):
            try:
                start = time.perf_counter()
                if name in ("s4d", "s5_onnx", "lru", "sgconv", "resnet1d"):
                    model, info = run_torch(name, data["root_train_csv"], data["root_val_csv"],
                                            data["root_test_csv"], classes, params,
                                            artifact_dir / f"{name}_trial{index}.onnx", seed + index, args.epochs)
                else:
                    model, info = tf_fit(name, data["root_train_csv"], data["root_val_csv"],
                                         classes, params, out / f"{name}_trial{index}.keras",
                                         seed + index, args.epochs)
                trials.append({"params": params, **info, "fit_time_s": time.perf_counter() - start})
            except Exception as exc:
                trials.append({"params": params, "failure_reason": repr(exc)})
        valid = [trial for trial in trials if "failure_reason" not in trial]
        if not valid:
            rows.append({"architecture": name, "failure_reason": "all trials failed", "trials": trials}); continue
        best = dict(max(valid, key=lambda item: (item.get("best_val_accuracy", -1), item.get("best_val_macro_f1", -1))))
        selected[name] = best
        # Frozen selection: retrain selected hyperparameters on train+validation, never test.
        try:
            start = time.perf_counter()
            if name in ("s4d", "s5_onnx", "lru", "sgconv", "resnet1d"):
                model, info = run_torch(name, data["root_train_val_csv"], data["root_val_csv"],
                                        data["root_test_csv"], classes, best["params"],
                                        artifact_dir / f"{name}.onnx", seed, args.epochs)
                artifact = Path(info["artifact"])
                adapter_name = name
            else:
                combined_x, combined_y = data["root_train_val_csv"]
                selected_epochs = int(best.get("epochs", args.epochs))
                model, info = tf_fit_final(
                    name,
                    (combined_x, combined_y),
                    classes,
                    best["params"],
                    out / f"{name}.keras",
                    seed,
                    selected_epochs,
                )
                adapter_name = name
                artifact = export_tf(model, artifact_dir / f"{name}.tflite")
            adapter = create_model(adapter_name)
            adapter.load(str(artifact))
            adapter.predict(data["root_test_csv"][0][0])
            pred = batch_predictions(name, model, data["root_test_csv"][0]).astype(np.int64)
            scores = metric(data["root_test_csv"][1], pred, classes)
            latencies = []
            for window in data["root_test_csv"][0][:min(100, len(pred))]:
                adapter.predict(window); latencies.append(adapter.latency_ms)
            scores.update({"architecture": name, "params": int(model.count_params() if hasattr(model, "count_params")
                         else sum(p.numel() for p in model.parameters())), "selected_config": best["params"],
                           "fit_time_s": time.perf_counter() - start, "artifact": str(artifact),
                           "artifact_latency_ms": float(np.median(latencies)), "trials": trials})
            selected[name]["deployment"] = scores; rows.append(scores)
        except Exception as exc:
            rows.append({"architecture": name, "selected_config": best["params"],
                         "failure_reason": repr(exc), "trials": trials})
    (out / "results.json").write_text(json.dumps({"data_source": {k: str(v) for k, v in data_paths.items()},
        "windowing": {"size": 156, "stride": 78, "features": config["runtime"]["feature_columns"]},
        "results": rows}, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in row.items() if k not in ("classwise", "confusion_matrix", "trials")}
                  for row in rows]).to_csv(out / "results.csv", index=False)
    classwise_rows = [
        {"architecture": row["architecture"], **item}
        for row in rows if "classwise" in row for item in row["classwise"]
    ]
    pd.DataFrame(classwise_rows).to_csv(out / "classwise.csv", index=False)
    confusion_rows = [
        {"architecture": row["architecture"], "true_class": classes[i],
         "predictions": values}
        for row in rows if "confusion_matrix" in row
        for i, values in enumerate(row["confusion_matrix"])
    ]
    pd.DataFrame(confusion_rows).to_json(out / "confusion_matrices.json",
                                         orient="records", indent=2)
    pd.DataFrame(confusion_rows).to_csv(out / "confusion_matrices.csv", index=False)
    # Robustness uses the already-windowed validation split, never test labels.
    generate_robustness(selected, data["root_val_csv"], classes, out)
    write_figures(rows, classes, out / "figures")
    write_report(rows, out, {key: len(value[0]) for key, value in data.items()})


def generate_robustness(selected: dict[str, Any], test: tuple[np.ndarray, np.ndarray],
                        classes: list[str], out: Path) -> None:
    """Write deterministic channel-dropout, jitter, missingness, noise, and resize checks."""
    rows = []
    for name, info in selected.items():
        deployment = info.get("deployment")
        if not deployment:
            continue
        try:
            adapter = create_model(name); adapter.load(deployment["artifact"])
            base = test[0][:100].copy()
            variants = {"channel_dropout": base * np.array([1, 1, 0, 1, 1, 1, 1]),
                        "clock_jitter": base + np.random.default_rng(42).normal(0, .01, base.shape),
                        "missing": np.where(np.random.default_rng(43).random(base.shape) < .1, 0, base),
                        "noise": base + np.random.default_rng(44).normal(0, .03, base.shape)}
            for variant, values in variants.items():
                pred = np.asarray([np.argmax(adapter.predict(window)) for window in values])
                rows.append({"architecture": name, "variant": variant,
                             **metric(test[1][:len(pred)], pred, classes)})
        except Exception as exc:
            rows.append({"architecture": name, "variant": "failure", "failure_reason": repr(exc)})
    pd.DataFrame(rows).to_json(out / "robustness.json", orient="records", indent=2)
    pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)


def write_figures(rows: list[dict[str, Any]], classes: list[str], out: Path) -> None:
    """Save confusion matrices and a macro-F1 comparison graph at 300 DPI."""
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    valid = [row for row in rows if "confusion_matrix" in row]
    for row in valid:
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.imshow(np.asarray(row["confusion_matrix"]), cmap="Blues")
        axis.set_title(row["architecture"]); axis.set_xlabel("Predicted"); axis.set_ylabel("True")
        axis.set_xticks(range(len(classes)), classes, rotation=45); axis.set_yticks(range(len(classes)), classes)
        figure.tight_layout(); figure.savefig(out / f"{row['architecture']}_confusion.png", dpi=300); plt.close(figure)
    if valid:
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar([row["architecture"] for row in valid], [row["macro_f1"] for row in valid])
        axis.set_ylabel("Test macro-F1"); axis.tick_params(axis="x", rotation=60)
        figure.tight_layout(); figure.savefig(out / "macro_f1_comparison.png", dpi=300); plt.close(figure)


def write_report(rows: list[dict[str, Any]], out: Path,
                 window_counts: dict[str, int] | None = None) -> None:
    """Create a compact DOCX report including the prior baseline when available."""
    try:
        from docx import Document
        document = Document(); document.add_heading("Neural models experiment", 0)
        document.add_paragraph(
            "Protocol: train/validation selection, train+validation retraining, "
            "adapter reload, and one test evaluation per final artifact."
        )
        document.add_paragraph(
            "Data source: train.csv, val.csv, train_val_combined.csv, and test.csv. "
            "Windows: 156 samples, stride 78, features Ax/Ay/Az/Gx/Gy/Gz/Speed."
        )
        if window_counts:
            document.add_paragraph(
                "Window counts: " + ", ".join(
                    f"{key}={value}" for key, value in window_counts.items()
                ) + "."
            )
        document.add_paragraph(
            "Limitations: TensorFlow recurrent artifacts use Keras fallback when "
            "TFLite conversion requires unsupported TensorList operations; PyTorch "
            "models are reported failed when ONNX export is unavailable. Results "
            "are not independent holdout measurements beyond the configured test split."
        )
        table = document.add_table(rows=1, cols=4)
        for cell, text in zip(table.rows[0].cells, ("Architecture", "Accuracy", "Macro-F1", "Status")):
            cell.text = text
        for row in rows:
            cells = table.add_row().cells
            cells[0].text = row["architecture"]; cells[1].text = str(row.get("accuracy", ""))
            cells[2].text = str(row.get("macro_f1", ""))
            cells[3].text = "failed: " + str(row.get("failure_reason", "")) if "failure_reason" in row else "ok"
        baseline = ROOT / "logs" / "rigorous_experiment" / "results.json"
        if baseline.exists():
            document.add_heading("Prior rigorous tree/ensemble baseline", 1)
            document.add_paragraph("Included unchanged for comparison; it was not retuned.")
            document.add_paragraph(baseline.read_text(encoding="utf-8")[:2000])
        report_path = out / "neural_models_experiment.docx"
        try:
            document.save(report_path)
        except PermissionError:
            document.save(out / "neural_models_experiment_updated.docx")
    except Exception as exc:
        (out / "report_failure.txt").write_text(repr(exc), encoding="utf-8")


if __name__ == "__main__":
    main()
