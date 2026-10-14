"""Build one Word report from the completed root-split experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
FIGURES = LOGS / "figures"
OUTPUT = ROOT / "logs" / "COMPREHENSIVE_MODEL_REPORT.docx"

MODEL_INFO: dict[str, tuple[str, str, str]] = {
    "histgradientboosting": ("Tree boosting", "Histogram gradient boosting over flattened windows.", "max_iter=100, random_state=42"),
    "lightgbm": ("Tree boosting", "LightGBM multiclass gradient boosting for tabular sensor windows.", "n_estimators=200, learning_rate=0.05, num_leaves=31, class_weight=balanced"),
    "random_forest": ("Tree ensemble", "Bagged decision trees for nonlinear sensor interactions.", "n_estimators=100, class_weight=balanced, n_jobs=-1"),
    "xgboost": ("Tree boosting", "Regularized gradient-boosted decision trees.", "n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8"),
    "logistic_regression": ("Linear baseline", "Linear multiclass classifier over flattened windows.", "max_iter=300, class_weight=balanced"),
    "linear_svm": ("Linear baseline", "Linear maximum-margin classifier over flattened windows.", "class_weight=balanced, max_iter=1000"),
    "mlp_tabular": ("Feed-forward neural network", "Two-hidden-layer neural network over flattened windows.", "Dense 128/64, dropout=0.2, Adam, epochs=10, batch_size=64"),
    "conv_transformer": ("Transformer", "Convolutional stem followed by transformer blocks.", "Existing project transformer builder, default configuration"),
    "gru": ("Recurrent neural network", "Gated recurrent sequence classifier.", "Existing GRU builder, default configuration"),
    "lstm": ("Recurrent neural network", "Long short-term memory sequence classifier.", "Existing LSTM builder, default configuration"),
    "fusformer": ("Transformer", "Fusion transformer with short and long temporal branches.", "Existing project transformer builder, default configuration"),
    "mega": ("Transformer", "Moving-average gated attention model.", "Existing project transformer builder, default configuration"),
    "convnet_har": ("Convolutional neural network", "Three-block temporal ConvNet with global pooling.", "Conv1D filters=64, kernels=7/5/3, epochs=10"),
    "tcnca_v3": ("Temporal convolution", "Dilated temporal convolution variant with channel attention.", "Existing TCNCA v3 builder, default configuration"),
    "rbf_svm": ("Kernel baseline", "RBF-kernel support-vector classifier.", "kernel=rbf, probability=True, class_weight=balanced"),
    "tcnca_v2": ("Temporal convolution", "TCNCA ablation with chunked attention.", "Existing TCNCA v2 builder, default configuration"),
    "tcnca_v1": ("Temporal convolution", "TCNCA ablation without the final attention variant.", "Existing TCNCA v1 builder, default configuration"),
    "tst": ("Transformer", "Temporal-spatial transformer with positional embeddings.", "d_model=24, heads=2, feedforward=48, blocks=2, dropout=0.1"),
    "s4d": ("State-space model", "Diagonal state-space sequence classifier.", "Existing S4D builder, default configuration"),
    "s5_onnx": ("State-space model", "ONNX S5 state-space sequence classifier.", "Existing S5 ONNX builder, default configuration"),
    "lru": ("State-space model", "Linear recurrent unit sequence classifier.", "Existing LRU builder, default configuration"),
    "sgconv": ("Structured convolution", "Depthwise and dilated convolution classifier.", "Existing SGConv builder, default configuration"),
    "resnet1d": ("Convolutional neural network", "Residual one-dimensional temporal convolution network.", "Residual blocks, Adam, epochs=10"),
    "egodrive_rt": ("Unavailable", "No trainable implementation or usable pretrained weights.", "Not applicable"),
    "egodrive_max": ("Unavailable", "No trainable implementation or usable pretrained weights.", "Not applicable"),
}


def shade(cell: object, fill: str) -> None:
    """Shade one Word table cell."""
    properties = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    properties.append(element)


def add_table(document: Document, frame: pd.DataFrame, columns: list[str], headers: list[str] | None = None) -> None:
    """Add a compact dataframe table."""
    if frame.empty:
        document.add_paragraph("No data available.")
        return
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    headers = headers or columns
    table = document.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = str(header)
        shade(cell, "D9EAF7")
        cell.paragraphs[0].runs[0].bold = True
    for _, row in frame[columns].iterrows():
        cells = table.add_row().cells
        for index, column in enumerate(columns):
            value = row[column]
            if isinstance(value, (float, np.floating)):
                value = f"{value:.4f}"
            cells[index].text = str(value)
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for run in cells[index].paragraphs[0].runs:
                run.font.size = Pt(7.5)
    document.add_paragraph()


def add_picture(document: Document, path: Path, caption: str, width: float = 6.2) -> None:
    """Embed an existing PNG figure when available."""
    if not path.exists():
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    caption_paragraph = document.add_paragraph(caption)
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.runs[0].italic = True
    caption_paragraph.runs[0].font.size = Pt(8)


def load_config() -> dict[str, Any]:
    """Load project configuration."""
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def build_report() -> Path:
    """Build and save the all-in-one root-split report."""
    config = load_config()
    metrics = pd.read_csv(LOGS / "comprehensive_metrics.csv")
    classwise = pd.read_csv(LOGS / "comprehensive_classwise.csv")
    robustness = pd.read_csv(LOGS / "comprehensive_robustness.csv")
    metadata = json.loads((LOGS / "comprehensive_workflow.json").read_text(encoding="utf-8"))
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.6)
    section.right_margin = Inches(0.6)
    document.styles["Normal"].font.name = "Aptos"
    document.styles["Normal"].font.size = Pt(9)
    document.styles["Title"].font.color.rgb = RGBColor(31, 78, 121)

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Two-Wheeler Driving Behaviour Recognition\nComprehensive Root-Split Model Report")
    subtitle = document.add_paragraph("Train / validation / combined retraining / held-out test evaluation")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    evaluated = metrics[metrics["status"] == "evaluated"].copy()
    best = evaluated.sort_values("test_macro_f1", ascending=False).iloc[0]
    validation_best = evaluated.sort_values("val_macro_f1", ascending=False).iloc[0]

    document.add_heading("1. Executive Summary", level=1)
    document.add_paragraph(
        f"{len(evaluated)} trainable models were evaluated using the supplied split files. "
        f"The highest validation macro-F1 was {validation_best['val_macro_f1']:.4f} ({validation_best['model']}); "
        f"the highest held-out test macro-F1 was {best['test_macro_f1']:.4f} ({best['model']}) with "
        f"{best['test_accuracy']:.2%} accuracy and {best['test_latency_mean_ms']:.2f} ms mean measured adapter latency."
    )
    document.add_paragraph("The final test set was not used for model selection. Test metrics come from reloaded saved deployment artifacts evaluated once.")

    document.add_heading("2. Dataset and Metrics", level=1)
    dataset = pd.DataFrame([
        ["Training source", "train.csv"],
        ["Validation source", "val.csv"],
        ["Final fitting source", "train_val_combined.csv"],
        ["Held-out test source", "test.csv"],
        ["Window counts", "train=3,875; val=1,045; combined=4,921; test=1,045"],
        ["Features", "Ax, Ay, Az, Gx, Gy, Gz, Speed"],
        ["Window / stride", "156 samples / 78 samples"],
        ["Classes", "BUMP, LEFT, RIGHT, STOP, STRAIGHT"],
        ["Metrics", "Accuracy, macro-F1, class-wise precision/recall/F1, latency"],
    ], columns=["Item", "Value"])
    add_table(document, dataset, ["Item", "Value"])
    document.add_paragraph("Windows and majority-vote labels were generated independently within each CSV; no window crosses a split boundary.")

    document.add_heading("3. Model Explanations and Hyperparameters", level=1)
    explanations = pd.DataFrame([
        {"model": name, "family": info[0], "explanation": info[1], "hyperparameters": info[2]}
        for name, info in MODEL_INFO.items()
    ])
    add_table(document, explanations, ["model", "family", "explanation", "hyperparameters"], ["Model", "Family", "What it does", "Hyperparameters"])

    document.add_heading("4. Metrics for Every Model", level=1)
    metric_columns = ["model", "status", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_latency_mean_ms", "test_latency_p95_ms", "test_windows_evaluated_once", "error"]
    add_table(document, metrics.sort_values("test_macro_f1", ascending=False, na_position="last"), metric_columns,
              ["Model", "Status", "Val accuracy", "Val macro-F1", "Test accuracy", "Test macro-F1", "Mean latency (ms)", "P95 latency (ms)", "Test windows", "Error"])

    document.add_heading("5. Benchmark Graphs", level=1)
    for filename, caption in [
        ("comprehensive_test_macro_f1.png", "Held-out test macro-F1 comparison"),
        ("comprehensive_test_latency_mean_ms.png", "Saved-artifact test latency comparison"),
        ("comprehensive_robustness.png", "Robustness comparison"),
    ]:
        add_picture(document, FIGURES / filename, caption)

    document.add_heading("6. Confusion Matrices and Class-Wise Results", level=1)
    add_table(document, classwise, ["model", "class", "precision", "recall", "f1-score", "support"],
              ["Model", "Class", "Precision", "Recall", "F1", "Support"])
    for path in sorted(FIGURES.glob("comprehensive_cm_*.png")):
        add_picture(document, path, f"Confusion matrix: {path.stem.replace('comprehensive_cm_', '').replace('_', ' ').title()}", width=5.4)

    document.add_heading("7. Robustness Channel Dropout", level=1)
    add_table(document, robustness[robustness["condition"] == "channel_dropout"], ["model", "condition", "accuracy", "macro_f1", "status"])
    document.add_heading("8. Robustness Clock Jitter", level=1)
    add_table(document, robustness[robustness["condition"] == "clock_jitter"], ["model", "condition", "accuracy", "macro_f1", "status"])
    document.add_heading("9. Robustness Missing Values", level=1)
    add_table(document, robustness[robustness["condition"] == "missing"], ["model", "condition", "accuracy", "macro_f1", "status"])
    document.add_heading("10. Robustness Noise", level=1)
    add_table(document, robustness[robustness["condition"] == "noise"], ["model", "condition", "accuracy", "macro_f1", "status"])
    document.add_heading("11. Robustness Window Size", level=1)
    add_table(document, robustness[robustness["condition"] == "window_size"], ["model", "condition", "accuracy", "macro_f1", "status"])

    document.add_heading("12. Final Comparison and Recommendation", level=1)
    comparison = evaluated.sort_values("test_macro_f1", ascending=False)[["model", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_latency_mean_ms", "test_latency_p95_ms"]]
    add_table(document, comparison, list(comparison.columns), ["Model", "Val macro-F1", "Test accuracy", "Test macro-F1", "Mean latency (ms)", "P95 latency (ms)"])
    document.add_paragraph(
        f"Recommendation: {validation_best['model']} was selected by validation macro-F1. "
        f"{best['model']} achieved the highest held-out test macro-F1. Final Raspberry Pi 5 deployment choice should consider both "
        "held-out macro-F1 and measured end-to-end latency, not adapter latency alone."
    )
    document.add_paragraph(f"Workflow metadata: {metadata}.")

    document.add_heading("13. Limitations", level=1)
    for limitation in [
        "Each model used its existing/default configuration; no separate hyperparameter sweep was performed in this run.",
        "The supplied CSV partitions are treated as fixed train, validation, combined, and test datasets; generalisation beyond their source population is not established.",
        "Robustness tests are simulated perturbations and do not replace field validation.",
        "Latency shown is adapter/model inference latency on the current machine, not Raspberry Pi 5 end-to-end latency.",
        "EgoDrive RT and EgoDrive Max have no available trainable implementation or usable pretrained weights and therefore have no claimed test metrics.",
    ]:
        document.add_paragraph(limitation, style="List Bullet")

    footer = document.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Two-Wheeler Driving Behaviour Recognition | Root-Split Comprehensive Report").font.size = Pt(8)
    document.save(OUTPUT)
    print(f"Saved Word report: {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build_report()
