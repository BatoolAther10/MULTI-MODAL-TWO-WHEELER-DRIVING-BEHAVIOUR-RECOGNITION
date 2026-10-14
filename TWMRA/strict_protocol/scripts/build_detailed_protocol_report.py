"""Build a detailed report for both evaluation protocols."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOGS = ROOT / "logs"
OUTPUT = LOGS / "DETAILED_PROTOCOL_REPORT.docx"
CLASSES = ["BUMP", "LEFT", "RIGHT", "STOP", "STRAIGHT"]

FAMILIES = {
    "histgradientboosting": "Tree boosting", "lightgbm": "Tree boosting",
    "rbf_svm": "Kernel SVM", "xgboost": "Tree boosting",
    "mlp_tabular": "Tabular neural", "random_forest": "Tree ensemble",
    "logistic_regression": "Linear baseline", "linear_svm": "Linear SVM",
    "tcn": "Temporal convolution", "tcnca_v1": "TCN/channel attention",
    "tcnca_v2": "TCN/channel attention", "tcnca_v3": "TCN/channel attention",
    "gru": "Recurrent", "lstm": "Recurrent", "lstm_attention": "Recurrent + attention",
    "bilstm": "Bidirectional recurrent", "bilstm_attention": "Bidirectional + attention",
    "conv_transformer": "Transformer", "fusformer": "Transformer",
    "mega": "Transformer", "tst": "Time Series Transformer",
    "convnet_har": "Temporal CNN", "s4d": "State-space", "s5_onnx": "State-space",
    "lru": "State-space", "sgconv": "Structured convolution", "resnet1d": "Residual CNN",
}


def add_table(document: Document, frame: pd.DataFrame, columns: list[str], widths: list[float] | None = None) -> None:
    """Add a dataframe table with compact numeric formatting."""
    table = document.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    for cell, column in zip(table.rows[0].cells, columns):
        cell.text = str(column)
    for _, row in frame[columns].iterrows():
        cells = table.add_row().cells
        for cell, column in zip(cells, columns):
            value = row[column]
            if isinstance(value, (float, np.floating)):
                if column in {"accuracy", "macro_f1", "precision", "recall", "f1"}:
                    value = f"{value:.2%}"
                else:
                    value = f"{value:.4f}"
            cell.text = str(value)
    document.add_paragraph()


def add_matrix(document: Document, matrix: np.ndarray, title: str) -> None:
    """Add a numeric confusion matrix table."""
    frame = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
    frame.insert(0, "true \\ predicted", frame.index)
    document.add_paragraph(title)
    add_table(document, frame.reset_index(drop=True), ["true \\ predicted", *CLASSES])


def plot_scores(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Plot ranked model scores."""
    ordered = frame.sort_values("accuracy")
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.barh(ordered["model"], ordered["accuracy"] * 100, color="#4472C4")
    axis.axvline(75, color="#C00000", linestyle="--", label="75%")
    axis.set(xlabel="Accuracy (%)", title=title)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def build_report() -> Path:
    """Build the detailed DOCX report."""
    strict = pd.read_csv(LOGS / "all_27_models_accuracy_comparison.csv")
    strict["family"] = strict["model"].map(FAMILIES).fillna("Other")
    strict = strict.sort_values(["accuracy", "macro_f1"], ascending=False).reset_index(drop=True)
    strict.insert(0, "rank", np.arange(1, len(strict) + 1))
    random = pd.DataFrame([
        ["TCNCA optimized", "Temporal convolution/channel attention", 0.7990430622, 0.7999794175, "clean.csv random stratified 80/20"],
        ["TST optimized", "Time Series Transformer", 0.7511961722, 0.7405400467, "clean.csv random stratified 80/20"],
    ], columns=["model", "family", "accuracy", "macro_f1", "protocol"])

    score_figure = LOGS / "figures" / "detailed_strict_27_accuracy.png"
    score_figure.parent.mkdir(parents=True, exist_ok=True)
    plot_scores(strict, score_figure, "Strict four-file 27-model comparison")

    random_cm = {
        "TCNCA optimized": np.array([[175, 13, 7, 5, 35], [13, 85, 5, 2, 10], [4, 0, 55, 1, 2], [8, 2, 5, 161, 12], [23, 17, 16, 8, 268]]),
        "TST optimized": np.array([[154, 26, 20, 1, 34], [7, 94, 5, 0, 9], [1, 1, 55, 0, 5], [3, 2, 12, 155, 16], [57, 40, 70, 9, 269]]),
    }
    strict_cm = {
        "HistGradientBoosting": np.array([[142, 17, 7, 1, 68], [2, 89, 5, 0, 19], [1, 1, 45, 0, 15], [5, 2, 0, 136, 45], [33, 21, 16, 6, 369]]),
        "TST": np.array([[172, 22, 20, 5, 16], [13, 90, 5, 2, 5], [5, 0, 55, 1, 1], [15, 2, 9, 152, 10], [118, 55, 71, 9, 192]]),
    }
    robustness = pd.read_csv(LOGS / "rigorous_experiment" / "robustness.csv")
    window = robustness[robustness["test"].str.startswith("window_size_")].copy()

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Two-Wheeler Behaviour Recognition\nDetailed Protocol and 27-Model Report")
    run.bold = True
    run.font.size = Pt(18)
    document.add_paragraph(
        "This report keeps the paper-style random 80/20 results separate from the required "
        "independent four-file evaluation."
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("1. Executive summary", level=1)
    document.add_paragraph(
        "The best available paper-style random-split result is TCNCA optimized at 79.90% accuracy "
        "and 80.00% macro-F1. The paper-style TST result is 75.12% accuracy and 74.05% macro-F1. "
        "Under the required strict protocol, the best individual model is HistGradientBoosting at "
        "74.74% accuracy and 73.25% macro-F1; the validation-selected ensemble reaches 75.02% accuracy."
    )
    document.add_paragraph(
        "The strict protocol is the authoritative deployment comparison because `test.csv` is not "
        "used for training or hyperparameter selection."
    )

    document.add_heading("2. Dataset and common settings", level=1)
    add_table(document, pd.DataFrame([
        ["Raw features", "Ax, Ay, Az, Gx, Gy, Gz, Speed"],
        ["Window size", "156 samples"],
        ["Stride", "78 samples"],
        ["Classes", "BUMP, LEFT, RIGHT, STOP, STRAIGHT"],
        ["Strict train windows", "3,875"],
        ["Strict validation windows", "1,045"],
        ["Strict combined windows", "4,921"],
        ["Strict test windows", "1,045"],
        ["CSV encoding", "UTF-8 with BOM support (utf-8-sig)"],
    ], columns=["Setting", "Value"]), ["Setting", "Value"])

    document.add_heading("3. Protocol definitions", level=1)
    add_table(document, pd.DataFrame([
        ["Random 80/20", "Window clean.csv; stratified random split with seed 42", "Reference/paper-style benchmark; possible neighboring-window overlap"],
        ["Strict four-file", "train.csv -> val.csv -> train_val_combined.csv -> test.csv", "Independent held-out test; authoritative comparison"],
    ], columns=["Protocol", "Procedure", "Interpretation"]), ["Protocol", "Procedure", "Interpretation"])

    document.add_heading("4. Best random 80/20 classification results", level=1)
    add_table(document, random, ["model", "family", "accuracy", "macro_f1", "protocol"])
    for name, matrix in random_cm.items():
        add_matrix(document, matrix, f"Confusion matrix: {name} (random 80/20)")
    document.add_paragraph(
        "Only the completed optimized TCNCA and TST random-split experiments are claimed here. "
        "The other 25 models do not have equivalent completed paper-style 80/20 measurements in the "
        "current artifact set, so they are not assigned invented random-split values."
    )

    document.add_heading("5. Complete 27-model strict classification results", level=1)
    add_table(document, strict, ["rank", "model", "family", "accuracy", "macro_f1", "status"])
    document.add_picture(str(score_figure), width=Inches(6.7))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for name, matrix in strict_cm.items():
        add_matrix(document, matrix, f"Confusion matrix: {name} (strict four-file)")

    document.add_heading("6. Strict class-wise metrics", level=1)
    classwise_path = LOGS / "rigorous_experiment" / "histgradientboosting_classwise.json"
    if classwise_path.exists():
        with classwise_path.open("r", encoding="utf-8") as handle:
            classwise = pd.DataFrame(json.load(handle)["classwise"])
        add_table(document, classwise, ["class", "precision", "recall", "support"])
    document.add_paragraph(
        "Class-wise metrics show why macro-F1 is reported alongside accuracy: the class counts are "
        "imbalanced, and aggregate accuracy can hide weaker minority-class performance."
    )

    document.add_heading("7. Robustness results", level=1)
    add_table(document, robustness, ["test", "accuracy", "macro_f1"])
    document.add_paragraph(
        "These tests use the strict test windows with simulated perturbations and are diagnostic "
        "only; they are not additional independent holdout evaluations."
    )

    document.add_heading("8. Window-size sensitivity", level=1)
    add_table(document, window, ["test", "accuracy", "macro_f1"])
    document.add_paragraph(
        "The official model input remains 156 samples. Alternative window sizes in this table are "
        "resampled to the fixed model input and therefore indicate sensitivity, not retrained-model "
        "comparisons."
    )

    document.add_heading("9. Limitations and deployment notes", level=1)
    document.add_paragraph(
        "Random 80/20 results can be optimistic when overlapping windows are split randomly. "
        "They reproduce the paper-style result but should not replace the strict independent test. "
        "The strict results use saved artifacts and a single final test evaluation. Raspberry Pi 5 "
        "latency, memory, and end-to-end queueing still require hardware validation."
    )
    document.add_paragraph(
        "Unavailable EgoDrive RT and EgoDrive Max implementations are excluded from the 27 measured "
        "classification rows because no usable trainable implementation or pretrained artifact was available."
    )
    document.save(OUTPUT)
    print(f"Saved {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build_report()
