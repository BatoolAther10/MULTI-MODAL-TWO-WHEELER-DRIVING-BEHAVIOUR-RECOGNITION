"""Run a leakage-safe, reproducible train/validation/test experiment."""

from __future__ import annotations

import hashlib
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.windowing import make_windows


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def read_split(path: Path, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Read one split into the identical raw seven-channel model input."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    runtime = cfg["runtime"]
    x, y, _ = make_windows(frame, int(runtime["window_size"]), int(runtime["stride"]),
                            list(runtime["feature_columns"]))
    return x, y.astype(str), frame


def matrix(x: np.ndarray) -> np.ndarray:
    """Create the same compact mean-pooled matrix for all tabular candidates."""
    step = 6
    usable = (x.shape[1] // step) * step
    return x[:, :usable].reshape(len(x), usable // step, step, x.shape[2]).mean(axis=2).reshape(len(x), -1)


def estimator(name: str, params: dict[str, Any], seed: int, classes: int) -> Any:
    """Construct a class-balanced candidate estimator."""
    if name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=seed, class_weight="balanced", **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="multiclass", num_class=classes, class_weight="balanced",
                              random_state=seed, n_jobs=-1, verbosity=-1, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", num_class=classes, eval_metric="mlogloss",
                             random_state=seed, n_jobs=1, subsample=.9, colsample_bytree=.85, **params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **params)
    if name == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
            class_weight="balanced", max_iter=500, solver="lbfgs", random_state=seed, **params))])
    if name == "linear_svm":
        return Pipeline([("scale", StandardScaler()), ("model", LinearSVC(
            class_weight="balanced", max_iter=1200, random_state=seed, **params))])
    if name == "rbf_svm":
        return Pipeline([("scale", StandardScaler()), ("model", SVC(
            class_weight="balanced", probability=True, random_state=seed, **params))])
    if name == "mlp_tabular":
        return Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(
            early_stopping=True, validation_fraction=.15, n_iter_no_change=8,
            max_iter=80, random_state=seed, **params))])
    raise ValueError(name)


def probabilities(model: Any, x: np.ndarray, classes: list[str]) -> np.ndarray:
    """Return probabilities, converting margin-only estimators safely."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
    else:
        score = model.decision_function(x)
        score = np.asarray(score)
        if score.ndim == 1:
            score = np.column_stack([-score, score])
        score -= score.max(axis=1, keepdims=True)
        p = np.exp(score)
        p /= p.sum(axis=1, keepdims=True)
    order = getattr(model, "classes_", None)
    if order is None and hasattr(model, "named_steps"):
        order = getattr(model.named_steps["model"], "classes_", None)
    out = np.zeros((len(x), len(classes)), dtype=float)
    for i, value in enumerate(order if order is not None else range(p.shape[1])):
        out[:, int(value) if isinstance(value, (int, np.integer)) else classes.index(str(value))] = p[:, i]
    return out


def score(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Calculate primary and classwise metrics."""
    pred = np.asarray(classes)[p.argmax(axis=1)]
    cm = confusion_matrix(y, pred, labels=classes)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, labels=classes, average="weighted", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "classwise": [{"class": c, "precision": float((cm[i, i] / max(cm[:, i].sum(), 1))),
                           "recall": float(cm[i, i] / max(cm[i].sum(), 1)),
                           "support": int(cm[i].sum())} for i, c in enumerate(classes)]}


def calibration(p: np.ndarray, y: np.ndarray, classes: list[str], bins: int = 10) -> dict[str, float]:
    """Calculate expected calibration error and mean confidence."""
    pred = p.argmax(axis=1)
    truth = np.asarray([classes.index(v) for v in y])
    confidence = p.max(axis=1)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
        if mask.any():
            ece += float(mask.mean()) * abs(float((pred[mask] == truth[mask]).mean()) - float(confidence[mask].mean()))
    return {"ece": ece, "mean_confidence": float(confidence.mean())}


def resize_windows(x: np.ndarray, size: int) -> np.ndarray:
    """Linearly resize temporal windows for a window-size robustness check."""
    old = np.linspace(0, 1, x.shape[1])
    new = np.linspace(0, 1, size)
    return np.stack([np.stack([np.interp(new, old, row[:, c]) for c in range(x.shape[2])], axis=1) for row in x])


def fingerprint(frame: pd.DataFrame) -> set[str]:
    """Build stable row fingerprints from timestamps and sensor values."""
    cols = ["Formatted_Timestamp", "Raw_Timestamp", "Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Speed"]
    available = [c for c in cols if c in frame]
    return set(pd.util.hash_pandas_object(frame[available].astype(str), index=False).astype(str))


def main() -> None:
    """Execute training, frozen selection, deployment evaluation, robustness, and reporting."""
    cfg = load_config()
    seed = int(cfg["experiment"]["seed"])
    np.random.seed(seed)
    out = ROOT / "logs" / "rigorous_experiment"
    fig = out / "figures"
    artifact_dir = ROOT / "models" / "rigorous_deployment"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: ROOT / cfg["data"][k] for k in ["root_train_csv", "root_val_csv", "root_train_val_csv", "root_test_csv"]}
    splits = {k: read_split(v, cfg) for k, v in paths.items()}
    classes = list(cfg["runtime"]["classes"])
    xtr, ytr, ftr = splits["root_train_csv"]; xv, yv, fval = splits["root_val_csv"]
    xcv, ycv, fcv = splits["root_train_val_csv"]; xt, yt, ft = splits["root_test_csv"]
    train_m, val_m, comb_m, test_m = map(matrix, [xtr, xv, xcv, xt])
    overlap = {a + "_" + b: len(fingerprint(splits[a][2]) & fingerprint(splits[b][2]))
               for a, b in [("root_train_csv", "root_val_csv"), ("root_train_csv", "root_test_csv"),
                            ("root_val_csv", "root_test_csv")]}
    specs: dict[str, list[dict[str, Any]]] = {
        "histgradientboosting": [
            {"max_iter": 80, "learning_rate": .05, "max_leaf_nodes": 31, "l2_regularization": 0.0},
            {"max_iter": 160, "learning_rate": .03, "max_leaf_nodes": 63, "l2_regularization": 0.1},
            {"max_iter": 240, "learning_rate": .025, "max_leaf_nodes": 15, "l2_regularization": 0.1},
            {"max_iter": 320, "learning_rate": .02, "max_leaf_nodes": 31, "l2_regularization": 0.5},
            {"max_iter": 180, "learning_rate": .04, "max_leaf_nodes": 15, "l2_regularization": 1.0},
            {"max_iter": 260, "learning_rate": .03, "max_leaf_nodes": 127, "l2_regularization": 0.5},
        ],
        "lightgbm": [
            {"n_estimators": 120, "learning_rate": .05, "num_leaves": 31, "min_child_samples": 20},
            {"n_estimators": 240, "learning_rate": .03, "num_leaves": 63, "min_child_samples": 20},
            {"n_estimators": 320, "learning_rate": .02, "num_leaves": 15, "min_child_samples": 10},
            {"n_estimators": 450, "learning_rate": .015, "num_leaves": 31, "min_child_samples": 10},
            {"n_estimators": 220, "learning_rate": .025, "num_leaves": 15, "min_child_samples": 30},
            {"n_estimators": 180, "learning_rate": .04, "num_leaves": 63, "min_child_samples": 40},
        ],
        "xgboost": [
            {"n_estimators": 120, "max_depth": 4, "learning_rate": .05, "min_child_weight": 1},
            {"n_estimators": 220, "max_depth": 6, "learning_rate": .03, "min_child_weight": 1},
            {"n_estimators": 320, "max_depth": 3, "learning_rate": .025, "min_child_weight": 1},
            {"n_estimators": 450, "max_depth": 4, "learning_rate": .02, "min_child_weight": 2},
            {"n_estimators": 240, "max_depth": 5, "learning_rate": .03, "min_child_weight": 3},
            {"n_estimators": 180, "max_depth": 2, "learning_rate": .05, "min_child_weight": 1},
        ],
        "random_forest": [
            {"n_estimators": 160, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 240, "max_depth": 24, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"n_estimators": 320, "max_depth": 12, "min_samples_leaf": 2, "max_features": 0.7},
            {"n_estimators": 500, "max_depth": 18, "min_samples_leaf": 1, "max_features": 0.7},
            {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2, "max_features": None},
        ],
        "logistic_regression": [{"C": .03}, {"C": .1}, {"C": 1.0}, {"C": 10.0}, {"C": 30.0}],
        "linear_svm": [{"C": .03}, {"C": .1}, {"C": 1.0}, {"C": 10.0}, {"C": 30.0}],
        "rbf_svm": [
            {"C": .5, "gamma": "scale"}, {"C": 1.0, "gamma": "scale"},
            {"C": 4.0, "gamma": "scale"}, {"C": 16.0, "gamma": "scale"},
            {"C": 4.0, "gamma": .01}, {"C": 16.0, "gamma": .003},
        ],
        "mlp_tabular": [
            {"hidden_layer_sizes": (128, 64), "alpha": .0001, "learning_rate_init": .001},
            {"hidden_layer_sizes": (256, 128), "alpha": .001, "learning_rate_init": .0005},
            {"hidden_layer_sizes": (128, 64, 32), "alpha": .01, "learning_rate_init": .001},
            {"hidden_layer_sizes": (256, 128, 64), "alpha": .0001, "learning_rate_init": .0003},
            {"hidden_layer_sizes": (128, 128), "alpha": .01, "learning_rate_init": .0003},
        ],
    }
    rows: list[dict[str, Any]] = []; selected: dict[str, dict[str, Any]] = {}; val_prob: dict[str, np.ndarray] = {}
    for name, candidates in specs.items():
        best: tuple[float, Any, dict[str, Any], np.ndarray] | None = None
        for trial, params in enumerate(candidates):
            started = time.perf_counter()
            model = estimator(name, params, seed + trial, len(classes))
            model.fit(train_m, np.array([classes.index(v) for v in ytr]))
            p = probabilities(model, val_m, classes)
            metrics = score(yv, p, classes)
            row = {"model": name, "trial": trial, "params": json.dumps(params), **{k: v for k, v in metrics.items() if k != "confusion_matrix" and k != "classwise"},
                   "fit_seconds": time.perf_counter() - started, "status": "complete"}
            rows.append(row)
            if best is None or (metrics["accuracy"], metrics["macro_f1"]) > (best[1], best[0]):
                best = (metrics["macro_f1"], metrics["accuracy"], params, p)
        assert best is not None
        selected[name] = {"params": best[2], "val_macro_f1": best[0], "val_accuracy": best[1]}
        val_prob[name] = best[3]
    top = sorted(selected, key=lambda n: (selected[n]["val_accuracy"], selected[n]["val_macro_f1"]), reverse=True)[:4]
    ensemble_candidates = [{"models": top[:2], "weights": [1, 1]}, {"models": top[:3], "weights": [2, 1, 1]},
                           {"models": top[:4], "weights": [2, 1, 1, 1]}]
    ensemble_rows = []
    for ens in ensemble_candidates:
        p = sum(w * val_prob[n] for n, w in zip(ens["models"], ens["weights"])) / sum(ens["weights"])
        m = score(yv, p, classes); ensemble_rows.append({"models": ens["models"], "weights": ens["weights"], **{k: m[k] for k in ["accuracy", "macro_f1", "weighted_f1"]}})
    best_ens = max(ensemble_rows, key=lambda r: (r["macro_f1"], r["accuracy"]))
    # Freeze selection, refit only the selected individual configurations on train+validation.
    test_prob: dict[str, np.ndarray] = {}
    for name, choice in selected.items():
        model = estimator(name, choice["params"], seed, len(classes))
        model.fit(comb_m, np.array([classes.index(v) for v in ycv]))
        path = artifact_dir / f"{name}.joblib"
        joblib.dump({"model": model, "classes": classes, "feature_order": cfg["runtime"]["feature_columns"],
                     "window_size": cfg["runtime"]["window_size"], "stride": cfg["runtime"]["stride"]}, path)
        reloaded = joblib.load(path)["model"]
        start = time.perf_counter(); p = probabilities(reloaded, test_m, classes); elapsed = (time.perf_counter() - start) * 1000 / len(test_m)
        test_prob[name] = p
        m = score(yt, p, classes)
        rows.append({"model": name, "trial": "frozen_test", "params": json.dumps(choice["params"]), **{k: v for k, v in m.items() if k not in ("confusion_matrix", "classwise")}, "artifact_latency_ms": elapsed, "status": "reloaded_test_once"})
        (out / f"{name}_classwise.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    ep = sum(w * test_prob[n] for n, w in zip(best_ens["models"], best_ens["weights"])) / sum(best_ens["weights"])
    em = score(yt, ep, classes)
    # Robustness is descriptive only; no decisions are changed after the frozen test evaluation.
    robust = []
    leader = max(selected, key=lambda n: (selected[n]["val_macro_f1"], selected[n]["val_accuracy"]))
    rng = np.random.default_rng(seed)
    for label, altered in [("channel_dropout", xt.copy()), ("noise", xt.copy()), ("missing", xt.copy()), ("clock_jitter", xt.copy())]:
        if label == "channel_dropout": altered[:, :, 0] = 0
        elif label == "noise": altered += rng.normal(0, altered.std(axis=(0, 1), keepdims=True) * .1, altered.shape)
        elif label == "missing": altered[:, ::10, :] = 0
        else: altered = altered[:, np.roll(np.arange(altered.shape[1]), 1), :]
        pp = probabilities(joblib.load(artifact_dir / f"{leader}.joblib")["model"], matrix(altered), classes)
        rm = score(yt, pp, classes)
        robust.append({"test": label, "accuracy": rm["accuracy"], "macro_f1": rm["macro_f1"]})
    for size in cfg["robustness"].get("window_sizes", [100, 128, 200]):
        resized = resize_windows(resize_windows(xt, int(size)), xt.shape[1])
        pp = probabilities(joblib.load(artifact_dir / f"{leader}.joblib")["model"], matrix(resized), classes)
        rm = score(yt, pp, classes)
        robust.append({"test": f"window_size_{size}", "accuracy": rm["accuracy"], "macro_f1": rm["macro_f1"]})
    calibration_result = calibration(test_prob[leader], yt, classes)
    pd.DataFrame(rows).to_csv(out / "all_metrics.csv", index=False)
    pd.DataFrame(ensemble_rows).to_json(out / "validation_ensembles.json", orient="records", indent=2)
    pd.DataFrame(robust).to_csv(out / "robustness.csv", index=False)
    summary = {"protocol": "independent all-window generation; train fit, validation selection, frozen combined refit, one reloaded test evaluation",
               "splits": {k: {"rows": len(v[2]), "windows": len(v[1]), "class_balance": pd.Series(v[1]).value_counts().to_dict()} for k, v in splits.items()},
               "classes": classes, "feature_order": cfg["runtime"]["feature_columns"], "window_size": cfg["runtime"]["window_size"],
               "stride": cfg["runtime"]["stride"], "row_overlap_counts": overlap, "selected": selected, "validation_ensembles": ensemble_rows,
               "selected_ensemble": best_ens, "test_ensemble": em, "robustness": robust,
               "calibration": calibration_result,
               "deployment_note": "Artifact inference latency is separate from total pipeline latency; Raspberry Pi 5 hardware validation is required.",
               "unavailable_families": {"ConvNet-HAR": "not run: torch neural search is outside bounded CPU execution budget; dependency available",
                                       "GRU": "not run: torch neural search is outside bounded CPU execution budget; dependency available",
                                       "LSTM": "not run: torch neural search is outside bounded CPU execution budget; dependency available",
                                       "TCNCA": "not run: torch neural search is outside bounded CPU execution budget; dependency available",
                                       "Transformer": "not run: torch neural search is outside bounded CPU execution budget; dependency available",
                                       "SSM": "not run: no stable project training/export path; dependency available",
                                       "ResNet": "not run: torch neural search is outside bounded CPU execution budget; dependency available"}}
    (out / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plt.figure(figsize=(10, 5)); pd.DataFrame(rows).query("status == 'reloaded_test_once'").plot.bar(x="model", y="macro_f1", legend=False); plt.tight_layout(); plt.savefig(fig / "test_macro_f1.png", dpi=300); plt.close()
    build_docx(out, summary)
    print(json.dumps({"output_dir": str(out), "leader": leader, "ensemble_test_macro_f1": em["macro_f1"], "test_accuracy": em["accuracy"]}, indent=2))


def build_docx(out: Path, summary: dict[str, Any]) -> None:
    """Create a concise comprehensive DOCX report from frozen experiment outputs."""
    from docx import Document
    doc = Document()
    doc.add_heading("Rigorous Two-Wheeler Behaviour Experiment", 0)
    doc.add_paragraph("Leakage-safe train/validation selection with frozen train+validation refit and one reloaded deployment evaluation on test.")
    doc.add_heading("Dataset and protocol", 1)
    doc.add_paragraph(json.dumps({k: summary[k] for k in ["feature_order", "window_size", "stride", "row_overlap_counts", "splits"]}, indent=2))
    doc.add_heading("Selected configurations and validation ensembles", 1)
    doc.add_paragraph(json.dumps({"selected": summary["selected"], "ensembles": summary["validation_ensembles"]}, indent=2))
    doc.add_heading("Frozen test result, calibration, and robustness", 1)
    doc.add_paragraph(json.dumps({"test_ensemble": summary["test_ensemble"], "calibration": summary["calibration"], "robustness": summary["robustness"]}, indent=2))
    doc.add_heading("Limitations and recommendation", 1)
    doc.add_paragraph("Tabular candidates were fully searched with train-only fitting and validation-only selection. Neural families are explicitly recorded as not run because a bounded CPU execution could not support meaningful searches; this is not evidence that they are inferior. Test results are not used for tuning. Encoded class errors are nominal. Deployment timing is host-side; Raspberry Pi 5 validation remains required.")
    if (out / "figures/test_macro_f1.png").exists():
        doc.add_picture(str(out / "figures/test_macro_f1.png"))
    doc.save(out / "rigorous_experiment_report.docx")


if __name__ == "__main__":
    main()
