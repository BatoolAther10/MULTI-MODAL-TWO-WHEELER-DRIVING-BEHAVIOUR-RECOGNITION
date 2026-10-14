"""Build a DOCX comparison of random-split and strict-split results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOGS = ROOT / "logs"
OUTPUT = LOGS / "PROTOCOL_COMPARISON_REPORT.docx"
CLASSES = ["BUMP", "LEFT", "RIGHT", "STOP", "STRAIGHT"]

from utils.windowing import make_windows


def load_tflite(path: Path, windows: np.ndarray) -> np.ndarray:
    """Predict a batch of windows with a TFLite artifact."""
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    predictions: list[int] = []
    for window in windows:
        interpreter.set_tensor(input_detail["index"], window[None].astype(np.float32))
        interpreter.invoke()
        predictions.append(int(np.argmax(interpreter.get_tensor(output_detail["index"])[0])))
    return np.asarray(predictions, dtype=np.int64)


def add_table(document: Document, frame: pd.DataFrame, columns: list[str]) -> None:
    """Add a dataframe as a compact Word table."""
    table = document.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    for cell, column in zip(table.rows[0].cells, columns):
        cell.text = str(column)
    for _, row in frame[columns].iterrows():
        cells = table.add_row().cells
        for cell, column in zip(cells, columns):
            value = row[column]
            if isinstance(value, (float, np.floating)):
                value = f"{value:.4f}"
            cell.text = str(value)
    document.add_paragraph()


def save_confusion_figure(name: str, matrix: np.ndarray, title: str) -> Path:
    """Save a labelled confusion-matrix heatmap."""
    path = LOGS / "figures" / f"protocol_{name}_confusion.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 4.3))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted", ylabel="True", title=title)
    for row in range(5):
        for column in range(5):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def build_report() -> Path:
    """Create the protocol comparison report and supporting figures."""
    random_metrics = pd.DataFrame([
        ["TCNCA optimized", 0.7990430622, 0.7999794175, "Random 80/20 windows"],
        ["TST optimized", 0.7511961722, 0.7405400467, "Random 80/20 windows"],
    ], columns=["model", "accuracy", "macro_f1", "protocol"])
    strict_metrics = pd.DataFrame([
        ["HistGradientBoosting", 0.7473684211, 0.7324650868, "train/val/combined/test"],
        ["Validation ensemble", 0.7502392344, 0.7340495841, "train/val/combined/test"],
        ["TCN", 0.7320574163, 0.7198005777, "train/val/combined/test"],
        ["TST", 0.6956937799, 0.6821989425, "train/val/combined/test"],
    ], columns=["model", "accuracy", "macro_f1", "protocol"])
    comparison = pd.concat([random_metrics, strict_metrics], ignore_index=True)

    clean = pd.read_csv(ROOT / "data" / "processed" / "clean.csv", encoding="utf-8-sig")
    random_x, random_y, _ = make_windows(clean, 156, 78)
    labels = {label: index for index, label in enumerate(CLASSES)}
    random_y_int = np.asarray([labels[str(value)] for value in random_y])
    _, random_test_x, _, random_test_y = train_test_split(
        random_x.astype(np.float32), random_y_int, test_size=0.2,
        random_state=42, stratify=random_y_int
    )
    matrices: dict[str, np.ndarray] = {}
    tcnca_path = ROOT / "models" / "tcnca_optimized.tflite"
    tst_random_path = ROOT / "models" / "tst_optimized.tflite"
    if tcnca_path.exists():
        matrices["80_20_tcnca"] = confusion_matrix(
            random_test_y, load_tflite(tcnca_path, random_test_x), labels=range(5)
        )
    if tst_random_path.exists():
        matrices["80_20_tst"] = confusion_matrix(
            random_test_y, load_tflite(tst_random_path, random_test_x), labels=range(5)
        )

    strict_x, strict_y, _ = make_windows(
        pd.read_csv(ROOT / "test.csv", encoding="utf-8-sig"), 156, 78,
        ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "Speed"]
    )
    strict_y_int = np.asarray([labels[str(value)] for value in strict_y])
    hgb_bundle = joblib.load(ROOT / "models" / "rigorous_deployment" / "histgradientboosting.joblib")
    hgb = hgb_bundle["model"]
    strict_matrix = strict_x[:, :strict_x.shape[1] // 6 * 6].reshape(
        len(strict_x), strict_x.shape[1] // 6, 6, strict_x.shape[2]
    ).mean(axis=2).reshape(len(strict_x), -1)
    strict_pred = hgb.predict(strict_matrix)
    matrices["strict_hgb"] = confusion_matrix(strict_y_int, strict_pred, labels=range(5))
    strict_tst_path = ROOT / "models" / "tst.tflite"
    if strict_tst_path.exists():
        matrices["strict_tst"] = confusion_matrix(
            strict_y_int, load_tflite(strict_tst_path, strict_x), labels=range(5)
        )

    confusion_paths = [
        save_confusion_figure(name, matrix, name.replace("_", " ").title())
        for name, matrix in matrices.items()
    ]
    window_robustness = pd.read_csv(LOGS / "rigorous_experiment" / "robustness.csv")
    window_robustness = window_robustness[window_robustness["test"].str.startswith("window_size_")].copy()
    window_robustness["window_size"] = window_robustness["test"].str.extract(r"(\d+)").astype(int)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(window_robustness["window_size"], window_robustness["accuracy"], marker="o", label="HGB")
    axis.plot(window_robustness["window_size"], window_robustness["macro_f1"], marker="s", label="HGB macro-F1")
    axis.set(xlabel="Evaluation window size (resampled to 156 for fixed model input)",
             ylabel="Score", title="Window-size sensitivity")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    window_figure = LOGS / "figures" / "protocol_window_sensitivity.png"
    figure.savefig(window_figure, dpi=300)
    plt.close(figure)

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Two-Wheeler Behaviour Recognition\nProtocol Comparison Report")
    run.bold = True
    run.font.size = Pt(18)
    document.add_paragraph(
        "This report separates the paper-style random 80/20 window split from the required "
        "train/validation/combined/independent-test protocol."
    )
    document.add_heading("1. Protocol summary", level=1)
    add_table(document, pd.DataFrame([
        ["Random 80/20", "Windows from clean.csv; stratified random split; seed 42", "Reference reproduction; overlapping windows may cross split"],
        ["Strict four-file", "train.csv -> val.csv -> train_val_combined.csv -> test.csv", "Authoritative independent-test comparison"],
    ], columns=["Protocol", "Procedure", "Interpretation"]), ["Protocol", "Procedure", "Interpretation"])
    document.add_heading("2. Best measured accuracies", level=1)
    comparison["accuracy_percent"] = comparison["accuracy"] * 100
    comparison["macro_f1_percent"] = comparison["macro_f1"] * 100
    add_table(document, comparison, ["protocol", "model", "accuracy_percent", "macro_f1_percent"])
    document.add_paragraph(
        "The 75.12% TST result is reproduced under the random 80/20 protocol. "
        "Under the required strict protocol, the best individual model is HistGradientBoosting "
        "at 74.74%; the validation-selected ensemble reaches 75.02%."
    )
    document.add_heading("3. Confusion matrices", level=1)
    for path in confusion_paths:
        document.add_picture(str(path), width=Inches(5.5))
        document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_heading("4. Window-size sensitivity", level=1)
    add_table(document, window_robustness, ["test", "accuracy", "macro_f1"])
    document.add_picture(str(window_figure), width=Inches(6.2))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(
        "The sensitivity check is descriptive: the trained tabular model requires 156-sample "
        "input, so alternative windows are linearly resampled to that input length. These values "
        "do not replace the official 156-sample test result and were not used for selection."
    )
    document.add_heading("5. Robustness", level=1)
    robustness = pd.read_csv(LOGS / "rigorous_experiment" / "robustness.csv")
    add_table(document, robustness, ["test", "accuracy", "macro_f1"])
    document.add_paragraph(
        "Robustness conditions are simulated channel dropout, noise, missing samples, clock jitter, "
        "and window-size changes. They are diagnostic and do not constitute additional independent holdout sets."
    )
    document.add_heading("6. Dataset and fixed settings", level=1)
    add_table(document, pd.DataFrame([
        ["Features", "Ax, Ay, Az, Gx, Gy, Gz, Speed"],
        ["Window", "156 samples"],
        ["Stride", "78 samples"],
        ["Classes", "BUMP, LEFT, RIGHT, STOP, STRAIGHT"],
        ["Strict split windows", "Train 3,875; validation 1,045; combined 4,921; test 1,045"],
    ], columns=["Setting", "Value"]), ["Setting", "Value"])
    document.add_heading("7. Limitations", level=1)
    document.add_paragraph(
        "Random-window results are not directly comparable to the strict independent test because "
        "overlapping neighboring windows can be distributed across both random partitions. "
        "Raspberry Pi 5 latency and memory still require hardware measurement."
    )
    document.save(OUTPUT)
    comparison.to_csv(LOGS / "protocol_comparison_metrics.csv", index=False)
    window_robustness.to_csv(LOGS / "protocol_window_sensitivity.csv", index=False)
    print(f"Saved {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build_report()
