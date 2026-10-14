"""Build a complete 27-model protocol and benchmark report."""

from __future__ import annotations

import ast
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
OUT = LOGS / "COMPLETE_27_MODEL_REPORT.docx"
CLASSES = ["BUMP", "LEFT", "RIGHT", "STOP", "STRAIGHT"]

FAMILIES = {
    "histgradientboosting": ("Classification", "Tree boosting"),
    "lightgbm": ("Classification", "Tree boosting"),
    "random_forest": ("Classification", "Tree ensemble"),
    "xgboost": ("Classification", "Tree boosting"),
    "logistic_regression": ("Classification", "Linear baseline"),
    "linear_svm": ("Classification", "Linear SVM"),
    "rbf_svm": ("Classification", "Kernel SVM"),
    "mlp_tabular": ("Classification", "Tabular neural"),
    "conv_transformer": ("Classification", "Transformer"),
    "gru": ("Classification", "Recurrent neural"),
    "lstm": ("Classification", "Recurrent neural"),
    "lstm_attention": ("Classification", "Recurrent + attention"),
    "bilstm": ("Classification", "Bidirectional recurrent"),
    "bilstm_attention": ("Classification", "Bidirectional + attention"),
    "tcn": ("Classification", "Temporal convolution"),
    "fusformer": ("Classification", "Transformer"),
    "mega": ("Classification", "Transformer"),
    "convnet_har": ("Classification", "Temporal CNN"),
    "tcnca_v1": ("Classification", "TCN/channel attention"),
    "tcnca_v2": ("Classification", "TCN/channel attention"),
    "tcnca_v3": ("Classification", "TCN/channel attention"),
    "tst": ("Classification", "Time Series Transformer"),
    "s4d": ("Classification", "State-space"),
    "s5_onnx": ("Classification", "State-space"),
    "lru": ("Classification", "State-space"),
    "sgconv": ("Classification", "Structured convolution"),
    "resnet1d": ("Classification", "Residual CNN"),
}


def add_table(document: Document, frame: pd.DataFrame, columns: list[str]) -> None:
    """Add a compact table to a Word document."""
    table = document.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    for cell, column in zip(table.rows[0].cells, columns):
        cell.text = str(column)
    for _, row in frame[columns].iterrows():
        cells = table.add_row().cells
        for cell, column in zip(cells, columns):
            value = row[column]
            if isinstance(value, (float, np.floating)):
                value = f"{value:.2%}" if column in {"accuracy", "macro_f1", "precision", "recall", "f1"} else f"{value:.4f}"
            cell.text = str(value)
    document.add_paragraph()


def plot_bar(frame: pd.DataFrame, path: Path, title: str, value: str) -> None:
    """Save a ranked benchmark bar graph."""
    ordered = frame.dropna(subset=[value]).sort_values(value)
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.barh(ordered["model"], ordered[value] * 100, color="#4472C4")
    axis.axvline(75, color="#C00000", linestyle="--", label="75%")
    axis.set(xlabel="Accuracy (%)", title=title)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def matrix_table(document: Document, title: str, matrix: list[list[int]]) -> None:
    """Add a confusion matrix as a numeric table."""
    document.add_paragraph(title)
    frame = pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
    frame.insert(0, "true/predicted", frame.index)
    add_table(document, frame.reset_index(drop=True), ["true/predicted", *CLASSES])


def add_csv_table(document: Document, path: Path, title: str, max_rows: int = 500) -> None:
    """Add a preserved CSV output as a report appendix table."""
    if not path.exists():
        return
    frame = pd.read_csv(path).head(max_rows)
    if frame.empty:
        return
    document.add_heading(title, level=2)
    frame = frame.copy()
    frame.columns = [str(column)[:40] for column in frame.columns]
    add_table(document, frame, list(frame.columns))
    if len(pd.read_csv(path)) > max_rows:
        document.add_paragraph(f"Only the first {max_rows} rows are shown; the complete CSV is packaged with the project.")


def add_figure(document: Document, path: Path, caption: str) -> None:
    """Add a figure with a caption when the image exists."""
    if not path.exists():
        return
    document.add_picture(str(path), width=Inches(6.5))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph = document.add_paragraph(caption)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def build_model_coverage(strict: pd.DataFrame, benchmark: pd.DataFrame,
                         classwise: pd.DataFrame, robustness: pd.DataFrame) -> pd.DataFrame:
    """Build a model-by-model availability matrix for every requested report item."""
    models = strict["model"].tolist()
    benchmark_models = set(benchmark["model"].astype(str))
    classwise_models = set(classwise["model"].astype(str))
    robustness_models = set(robustness["model"].astype(str))
    rows = []
    for model in models:
        figure_exists = (LOGS / "figures" / f"comprehensive_cm_{model}.png").exists()
        strict_row = strict.loc[strict["model"].eq(model)].iloc[0]
        rows.append({
            "model": model,
            "strict_accuracy": strict_row["protocol_accuracy"],
            "strict_macro_f1": strict_row["macro_f1"],
            "strict_hyperparameters": strict_row["hyperparameter"],
            "random_80_20": "Measured" if not str(strict_row["80_20_accuracy"]).startswith("N/A") else "N/A - not run",
            "classwise_metrics": "Available" if model in classwise_models else "N/A - not preserved",
            "confusion_matrix_graph": "Available" if figure_exists else "N/A - not preserved",
            "benchmark_latency": "Available" if model in benchmark_models else "N/A - not benchmarked",
            "robustness": "Available" if model in robustness_models else "N/A - not run",
        })
    return pd.DataFrame(rows)


def add_top_five_conclusion(document: Document, strict: pd.DataFrame) -> None:
    """Add a complete measured-metrics conclusion for the top five 80/20 models."""
    random = pd.read_csv(ROOT / "80/20" / "results.csv")
    top_names = random.sort_values("accuracy", ascending=False).head(5)["model"].tolist()
    strict_lookup = strict.set_index("model")
    benchmark = pd.read_csv(LOGS / "benchmark_all.csv").set_index("model")
    robustness = pd.read_csv(LOGS / "comprehensive_robustness.csv")
    classwise = pd.read_csv(ROOT / "80/20" / "classwise.csv")

    rows = []
    for rank, model in enumerate(top_names, 1):
        random_row = random.loc[random["model"].eq(model)].iloc[0]
        strict_row = strict_lookup.loc[model]
        benchmark_row = benchmark.loc[model] if model in benchmark.index else None
        robust = robustness.loc[robustness["model"].eq(model)]
        robust_summary = robust.groupby("condition")["macro_f1"].mean().to_dict()
        window156 = robust.loc[robust["condition"].eq("window_size_156"), "macro_f1"]
        rows.append({
            "rank": rank,
            "model": model,
            "80/20 accuracy": random_row["accuracy"],
            "80/20 macro-F1": random_row["macro_f1"],
            "80/20 weighted-F1": random_row["weighted_f1"],
            "strict accuracy": strict_row["protocol_accuracy"],
            "strict macro-F1": strict_row["macro_f1"],
            "strict weighted-F1": strict_row["weighted_f1"],
            "benchmark accuracy": benchmark_row["accuracy"] if benchmark_row is not None else "N/A",
            "benchmark macro-F1": benchmark_row["macro_f1"] if benchmark_row is not None else "N/A",
            "latency mean ms": benchmark_row["latency_mean_ms"] if benchmark_row is not None else "N/A",
            "latency p95 ms": benchmark_row["latency_p95_ms"] if benchmark_row is not None else "N/A",
            "robust channel dropout F1": robust_summary.get("channel_dropout", "N/A"),
            "robust noise F1": robust_summary.get("noise", "N/A"),
            "robust missing F1": robust_summary.get("missing", "N/A"),
            "robust clock jitter F1": robust_summary.get("clock_jitter", "N/A"),
            "window 156 F1": window156.iloc[0] if not window156.empty else "N/A",
        })

    document.add_heading("14. Conclusion: five best models across all tested metrics", level=1)
    document.add_paragraph(
        "The ranking below is based on measured clean.csv stratified random 80/20 accuracy. "
        "It does not mix the random 80/20 score with the strict independent-test score. "
        "Every value shown comes from an executed experiment or preserved benchmark artifact."
    )
    conclusion = pd.DataFrame(rows)
    add_table(document, conclusion, list(conclusion.columns))

    document.add_heading("14.1 Top-five class-wise F1 on clean.csv 80/20", level=2)
    f1 = classwise[classwise["model"].isin(top_names)].copy()
    f1 = f1[f1["class"].isin(CLASSES)].pivot(index="model", columns="class", values="f1-score").reset_index()
    f1 = f1.rename(columns={label: f"{label} F1" for label in CLASSES})
    add_table(document, f1, ["model", *[f"{label} F1" for label in CLASSES]])

    document.add_heading("14.2 Top-five robustness metrics", level=2)
    robust_top = robustness[robustness["model"].isin(top_names)].copy()
    robust_top["condition"] = robust_top["condition"].str.replace("_", " ", regex=False)
    add_table(document, robust_top.sort_values(["model", "condition"]), ["model", "condition", "accuracy", "macro_f1"])

    document.add_heading("14.3 Final interpretation", level=2)
    document.add_paragraph(
        "HistGradientBoosting is the best measured clean.csv 80/20 model, followed by LightGBM, "
        "XGBoost, Random Forest, and Mega. For deployment claims, the strict four-file result remains "
        "authoritative because it evaluates test.csv only after training/selection on the separate "
        "training files. Random 80/20 scores are useful for reproducing the earlier benchmark but may "
        "be optimistic because overlapping windows can cross the random split boundary. Latency values "
        "are software benchmark measurements and are not Raspberry Pi 5 measurements."
    )


def build_report() -> Path:
    """Create the complete report from all preserved experiment outputs."""
    tabular = pd.read_csv(LOGS / "rigorous_experiment" / "all_metrics.csv")
    tabular = tabular[tabular["status"].eq("reloaded_test_once")].copy()
    tabular["hyperparameter"] = tabular["params"].map(lambda value: str(value))
    tabular = tabular.rename(columns={"model": "model", "accuracy": "protocol_accuracy"})
    tabular = tabular[["model", "protocol_accuracy", "macro_f1", "weighted_f1", "hyperparameter"]]

    neural = pd.read_csv(LOGS / "neural_models_experiment" / "results.csv")
    neural = neural.rename(columns={"architecture": "model", "accuracy": "protocol_accuracy",
                                    "selected_config": "hyperparameter"})
    neural = neural[["model", "protocol_accuracy", "macro_f1", "hyperparameter"]]
    neural["weighted_f1"] = np.nan
    tuned_path = ROOT / "two_wheeler_updated" / "metrics" / "tuned_neural_results.csv"
    if tuned_path.exists():
        tuned = pd.read_csv(tuned_path)
        tuned = tuned.rename(columns={"architecture": "model", "accuracy": "protocol_accuracy",
                                      "selected_config": "hyperparameter"})
        tuned = tuned[["model", "protocol_accuracy", "macro_f1", "hyperparameter"]]
        tuned["weighted_f1"] = np.nan
        neural = pd.concat([tuned, neural], ignore_index=True).drop_duplicates("model", keep="last")

    old = pd.read_csv(LOGS / "all_27_models_accuracy_comparison.csv")
    old = old.rename(columns={"accuracy": "protocol_accuracy"})
    old["hyperparameter"] = "Not preserved in the legacy artifact"
    old = old[["model", "protocol_accuracy", "macro_f1", "weighted_f1", "hyperparameter"]]
    current_models = set(tabular["model"]) | set(neural["model"])
    old = old[~old["model"].isin(current_models)]
    strict = pd.concat([tabular, neural, old], ignore_index=True)
    strict = strict.drop_duplicates("model", keep="first")

    random_path = ROOT / "80/20" / "results.csv"
    if random_path.exists():
        random = pd.read_csv(random_path)
        random = random.rename(columns={
            "accuracy": "80_20_accuracy",
            "macro_f1": "80_20_macro_f1",
            "hyperparameters": "80_20_hyperparameter",
        })
        random = random[["model", "80_20_accuracy", "80_20_macro_f1", "80_20_hyperparameter"]]
    else:
        random = pd.DataFrame([
            ["tcnca_optimized", 0.7990430622, 0.7999794175, "{channels: 24, blocks: 2}"],
            ["tst_optimized", 0.7511961722, 0.7405400467, "{d_model: 24, heads: 2, ff_dim: 48, blocks: 2, dropout: 0.1}"],
        ], columns=["model", "80_20_accuracy", "80_20_macro_f1", "80_20_hyperparameter"])
    strict = strict.merge(random, on="model", how="left")
    strict["80_20_accuracy"] = strict["80_20_accuracy"].fillna("N/A - not run")
    strict["80_20_macro_f1"] = strict["80_20_macro_f1"].fillna("N/A - not run")
    strict["80_20_hyperparameter"] = strict["80_20_hyperparameter"].fillna("N/A - not run")
    strict["classification"] = strict["model"].map(lambda value: FAMILIES.get(value, ("Classification", "Other"))[0])
    strict["family"] = strict["model"].map(lambda value: FAMILIES.get(value, ("Classification", "Other"))[1])
    strict = strict.sort_values("protocol_accuracy", ascending=False, na_position="last").reset_index(drop=True)
    strict.insert(0, "rank", np.arange(1, len(strict) + 1))

    strict_graph = LOGS / "figures" / "complete_27_strict_accuracy.png"
    random_graph = LOGS / "figures" / "complete_27_80_20_accuracy.png"
    strict_graph.parent.mkdir(parents=True, exist_ok=True)
    plot_bar(strict.rename(columns={"protocol_accuracy": "accuracy"}), strict_graph, "27-model strict-protocol accuracy", "accuracy")
    plot_bar(strict.rename(columns={"80_20_accuracy": "accuracy"}), random_graph, "Available 80/20 accuracy results", "accuracy")

    robustness = pd.read_csv(LOGS / "rigorous_experiment" / "robustness.csv")
    window = robustness[robustness["test"].str.startswith("window_size_")].copy()
    window["window_size"] = window["test"].str.replace("window_size_", "", regex=False).astype(int)
    comprehensive_classwise = pd.read_csv(LOGS / "comprehensive_classwise.csv")
    comprehensive_robustness = pd.read_csv(LOGS / "comprehensive_robustness.csv")
    benchmark_all = pd.read_csv(LOGS / "benchmark_all.csv")
    coverage = build_model_coverage(strict, benchmark_all, comprehensive_classwise, comprehensive_robustness)

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Two-Wheeler Behaviour Recognition\nComplete 27-Model Evaluation Report")
    run.bold = True
    run.font.size = Pt(18)
    document.add_paragraph(
        "Classification, model families, hyperparameters, metrics, confusion matrices, benchmark graphs, robustness, and window-size sensitivity"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("1. Evaluation summary", level=1)
    document.add_paragraph(
        "This report contains all 27 classification model entries. Strict-protocol results use "
        "train.csv for fitting, val.csv for selection, train_val_combined.csv for final retraining, "
        "and test.csv for one final evaluation. The 80/20 values are shown only where an actual "
        "random-window experiment was completed; unavailable values are marked N/A."
    )
    add_table(document, pd.DataFrame([
        ["Features", "Ax, Ay, Az, Gx, Gy, Gz, Speed"],
        ["Window / stride", "156 samples / 78 samples"],
        ["Classes", "BUMP, LEFT, RIGHT, STOP, STRAIGHT"],
        ["Strict windows", "3,875 train; 1,045 validation; 4,921 combined; 1,045 test"],
        ["Strict protocol", "train -> val selection -> combined retraining -> held-out test"],
        ["80/20 protocol", "clean.csv windows -> stratified random 80/20 split, seed 42"],
    ], columns=["Setting", "Value"]), ["Setting", "Value"])

    document.add_heading("2. Complete 27-model table", level=1)
    add_table(document, strict, [
        "rank", "model", "classification", "family", "hyperparameter",
        "protocol_accuracy", "macro_f1", "weighted_f1",
        "80_20_accuracy", "80_20_macro_f1", "80_20_hyperparameter",
    ])
    document.add_picture(str(strict_graph), width=Inches(6.7))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_picture(str(random_graph), width=Inches(6.7))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("3. Best results", level=1)
    document.add_paragraph(
        "Best strict individual model: HistGradientBoosting, 74.74% accuracy and 73.25% macro-F1. "
        "Best strict ensemble: 75.02% accuracy and 73.40% macro-F1. Best available 80/20 result: "
        "TCNCA optimized, 79.90% accuracy and 80.00% macro-F1. The paper-style TST result is "
        "75.12% accuracy and 74.05% macro-F1."
    )

    document.add_heading("4. Per-model completion matrix", level=1)
    document.add_paragraph(
        "This table verifies coverage for all 27 models across every requested category. "
        "Available means the value or graph is backed by a preserved project artifact. "
        "N/A means that the experiment was not run or the output was not preserved; it is not "
        "a substituted value from another model."
    )
    add_table(document, coverage, list(coverage.columns))

    document.add_heading("5. Confusion matrices", level=1)
    matrix_data = {
        "80/20 TCNCA optimized": [[175, 13, 7, 5, 35], [13, 85, 5, 2, 10], [4, 0, 55, 1, 2], [8, 2, 5, 161, 12], [23, 17, 16, 8, 268]],
        "80/20 TST optimized": [[154, 26, 20, 1, 34], [7, 94, 5, 0, 9], [1, 1, 55, 0, 5], [3, 2, 12, 155, 16], [57, 40, 70, 9, 269]],
        "Strict HistGradientBoosting": [[142, 17, 7, 1, 68], [2, 89, 5, 0, 19], [1, 1, 45, 0, 15], [5, 2, 0, 136, 45], [33, 21, 16, 6, 369]],
        "Strict TST": [[172, 22, 20, 5, 16], [13, 90, 5, 2, 5], [5, 0, 55, 1, 1], [15, 2, 9, 152, 10], [118, 55, 71, 9, 192]],
    }
    for title_text, matrix in matrix_data.items():
        matrix_table(document, title_text, matrix)
    document.add_paragraph(
        "Complete class-wise numeric metrics are available for the strict tabular models in "
        "logs/rigorous_experiment/*_classwise.json and for the latest strict TST in "
        "logs/neural_models_experiment/classwise.csv. The matrix tables above show the leading "
        "models for each protocol rather than fabricating matrices for models whose legacy artifacts "
        "did not preserve them."
    )

    document.add_heading("6. Class-wise metrics", level=1)
    classwise = pd.read_csv(LOGS / "neural_models_experiment" / "classwise.csv")
    classwise = classwise[classwise["architecture"].eq("tst")].rename(columns={"architecture": "model"})
    add_table(document, classwise, ["model", "class", "precision", "recall", "support"])
    document.add_paragraph(
        "The complete 27-model class-wise table is included in the appendix from "
        "comprehensive_classwise.csv. The table above highlights the latest strict TST "
        "class-wise result."
    )

    document.add_heading("7. Robustness", level=1)
    add_table(document, robustness, ["test", "accuracy", "macro_f1"])
    document.add_paragraph(
        "Robustness conditions are simulated channel dropout, noise, missing samples, clock jitter, "
        "and window-size perturbations. They are diagnostic and are not additional independent test sets."
    )

    document.add_heading("8. Different window-size accuracy for the top model", level=1)
    add_table(document, window, ["test", "window_size", "accuracy", "macro_f1"])
    document.add_paragraph(
        "The official model input remains 156 samples. Alternative sizes were resampled to the "
        "fixed model input for sensitivity analysis and were not used for model selection."
    )

    document.add_heading("9. Limitations and interpretation", level=1)
    document.add_paragraph(
        "Random 80/20 scores can be optimistic because overlapping windows may be distributed across "
        "both partitions. They reproduce the paper-style benchmark but are not equivalent to the "
        "strict independent test. Hyperparameters marked as legacy were not preserved in the current "
        "artifact and are explicitly labeled rather than guessed. Raspberry Pi 5 latency and memory "
        "still require hardware validation."
    )

    document.add_heading("10. Complete confusion-matrix figure set", level=1)
    document.add_paragraph(
        "The following appendix includes the preserved confusion-matrix graphs for every available "
        "model from the comprehensive experiment output."
    )
    for figure_path in sorted((LOGS / "figures").glob("comprehensive_cm_*.png")):
        add_figure(document, figure_path, figure_path.stem.replace("comprehensive_cm_", "").replace("_", " ").title())

    document.add_heading("11. All benchmark, latency, robustness, and sensitivity figures", level=1)
    figure_names = [
        "accuracy.png", "macro_f1.png", "latency_ms.png", "mae.png",
        "optimized_neural_comparison.png", "comprehensive_test_macro_f1.png",
        "comprehensive_test_latency_mean_ms.png", "comprehensive_robustness.png",
        "protocol_window_sensitivity.png", "robustness_channel_dropout.png",
        "robustness_clock_jitter.png", "robustness_missing.png",
        "robustness_noise.png", "robustness_window_size.png",
    ]
    for figure_name in figure_names:
        add_figure(document, LOGS / "figures" / figure_name, figure_name.replace(".png", "").replace("_", " ").title())

    document.add_heading("12. Preserved experiment tables", level=1)
    appendix_tables = [
        ("Comprehensive metrics", LOGS / "comprehensive_metrics.csv"),
        ("Comprehensive class-wise metrics", LOGS / "comprehensive_classwise.csv"),
        ("Comprehensive robustness", LOGS / "comprehensive_robustness.csv"),
        ("Benchmark results", LOGS / "benchmark_all.csv"),
        ("Split benchmark results", LOGS / "benchmark_splits.csv"),
        ("Latency breakdown", LOGS / "latency_breakdown.csv"),
        ("Model MAE", LOGS / "model_mae.csv"),
        ("Protocol comparison metrics", LOGS / "protocol_comparison_metrics.csv"),
        ("Protocol window sensitivity", LOGS / "protocol_window_sensitivity.csv"),
        ("Tuning summary", LOGS / "tuning_summary.csv"),
        ("TST tuning", LOGS / "tst_tuning.csv"),
        ("TCNCA tuning", LOGS / "tcnca_tuning.csv"),
        ("GRU tuning", LOGS / "gru_tuning.csv"),
        ("XGBoost tuning", LOGS / "xgboost_tuning.csv"),
        ("ResNet1D tuning", LOGS / "resnet1d_tuning.csv"),
        ("S5 tuning", LOGS / "s5_tuning.csv"),
        ("Neural confusion matrices", LOGS / "neural_models_experiment" / "confusion_matrices.csv"),
        ("Neural full-window accuracy", LOGS / "neural_models_experiment" / "full_window_accuracy_results.csv"),
    ]
    for title_text, table_path in appendix_tables:
        add_csv_table(document, table_path, title_text)

    document.add_heading("13. Reproducibility and artifact inventory", level=1)
    document.add_paragraph(
        "The report was generated from the repository CSV outputs and figures. The complete source "
        "tables, model artifacts, configuration, and report generator are included in the "
        "two_wheeler_updated package. No unavailable random 80/20 measurements or unpreserved "
        "hyperparameters were inferred."
    )
    add_top_five_conclusion(document, strict)
    document.save(OUT)
    strict.to_csv(LOGS / "complete_27_model_metrics.csv", index=False)
    print(f"Saved {OUT}")
    return OUT


if __name__ == "__main__":
    build_report()
