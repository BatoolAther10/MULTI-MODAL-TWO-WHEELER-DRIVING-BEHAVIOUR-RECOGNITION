"""Generate a consolidated Markdown report from experiment CSV artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def load_config() -> dict[str, Any]:
    """Load the project configuration."""
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv(name: str) -> pd.DataFrame:
    """Read one log CSV or return an empty dataframe."""
    path = LOGS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int | None = None) -> str:
    """Render a dataframe as a compact Markdown table."""
    if frame.empty:
        return "_No artifact available._"
    selected = frame[columns] if columns else frame
    if limit is not None:
        selected = selected.head(limit)
    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in selected.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def relative_figure(path: str) -> str:
    """Return a report-relative figure link."""
    return f"![{Path(path).stem}](logs/figures/{Path(path).name})"


def model_inventory() -> pd.DataFrame:
    """Build the implemented model inventory from benchmark artifacts."""
    benchmark = read_csv("benchmark_all.csv")
    if benchmark.empty:
        return pd.DataFrame(columns=["model", "status", "accuracy", "macro_f1"])
    return benchmark[["model", "status", "accuracy", "macro_f1"]].copy()


def generate_report() -> Path:
    """Generate REPORT.md from all available experiment outputs."""
    config = load_config()
    benchmark = read_csv("benchmark_all.csv")
    split = read_csv("benchmark_splits.csv")
    classwise = read_csv("classwise_all.csv")
    latency = read_csv("latency_breakdown.csv")
    tcnca = read_csv("benchmark_transformers.csv")
    inventory = model_inventory()

    lines = [
        "# Two-Wheeler Driving Behaviour Recognition Report",
        "",
        "## 1. Sampling Rate and Window Duration",
        "",
        f"Window configuration: {config['runtime']['window_size']} samples with stride {config['runtime']['stride']} at a nominal {config['runtime']['sensor_sample_rate_hz']} Hz.",
        "",
        "The processed artifacts are `data/processed/clean.csv` and `data/processed/windows.npz`. Sampling-rate audit output should be stored in `logs/audit_sample_rate.csv` when Task 1 audit execution is added.",
        "",
        "## 2. Model Inventory",
        "",
        markdown_table(inventory, ["model", "status", "accuracy", "macro_f1"]),
        "",
        "DTAAD is anomaly-only and is intentionally excluded from classification ranking. EgoDriveRT is unavailable because weights are not public; EgoDriveMax is rejected for deployment constraints.",
        "",
        "## 3. Full Benchmark",
        "",
        markdown_table(benchmark, ["model", "status", "accuracy", "macro_f1", "params", "latency_mean_ms", "latency_p95_ms", "benchmark_source", "window_count"]),
        "",
        "The unified benchmark used 500 windows from the full-window fallback because the random test split contains only 209 windows. These results are not an independent holdout estimate.",
        "",
        "## 4. Random versus Session-Wise Splits",
        "",
        markdown_table(split, ["model", "split_type", "train_sessions", "test_sessions", "accuracy", "macro_f1"]),
        "",
        "These comparisons reuse artifacts trained on the random split. A clean generalisation claim requires retraining separately for each session-wise training fold.",
        "",
        "## 5. Confusion Matrices and Class-Wise Metrics",
        "",
        markdown_table(classwise, ["model", "class", "precision", "recall", "f1", "support"]),
        "",
        "Generated confusion matrices:",
        "",
    ]

    for figure in sorted((LOGS / "figures").glob("cm_*.png")):
        lines.append(relative_figure(str(figure)))
    lines.extend([
        "",
        "## 6. TCNCA Ablation",
        "",
        markdown_table(tcnca, ["model", "accuracy", "macro_f1", "latency_ms", "params"]),
        "",
        "TCNCA v3 is marked as the final ablation variant. ConvTransformer achieved the strongest transformer-family score in the current benchmark, while v3 remains the requested TCNCA final result.",
        "",
        "## 7. End-to-End Latency Breakdown",
        "",
        markdown_table(latency, ["stage", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "iterations"]),
        "",
        "Model inference latency measures only adapter execution; total end-to-end latency includes queue, synchronization, feature extraction, fallback checking, and buffered CSV work.",
        "",
        "## 8. Robustness Results",
        "",
    ])

    robustness_files = sorted(LOGS.glob("robustness_*.csv"))
    for path in robustness_files:
        frame = pd.read_csv(path)
        lines.extend([f"### {path.stem.replace('_', ' ').title()}", "", markdown_table(frame), ""])
        figure = ROOT / "logs" / "figures" / f"{path.stem}.png"
        if figure.exists():
            lines.extend([relative_figure(str(figure)), ""])

    best = benchmark[benchmark["status"] == "benchmarked"].sort_values("macro_f1", ascending=False).iloc[0] if not benchmark.empty else None
    lines.extend([
        "## 9. Model Selection Recommendation",
        "",
    ])
    if best is not None:
        lines.append(f"The highest measured macro F1 is {best['model']} at {best['macro_f1']:.4f}. XGBoost remains a practical deployment candidate because its measured p95 latency is {benchmark.loc[benchmark['model'].eq('xgboost'), 'latency_p95_ms'].iloc[0]:.3f} ms in the 500-window benchmark and it is already integrated into the runtime path.")
    lines.extend([
        "Neural models should be selected only after Raspberry Pi measurements, because host-machine latency and TensorFlow Lite delegate behavior can differ materially from ARM64 deployment.",
        "",
        "## 10. Limitations",
        "",
        "- Only two sessions are available, so generalisation to unseen rides or riders cannot be claimed.",
        "- Session-wise comparisons reuse randomly trained artifacts and are therefore diagnostic rather than clean cross-session validation.",
        "- Transition labels, where used in future experiments, are heuristic majority-coverage labels.",
        "- Clock-jitter testing is simulated and does not measure real field clock drift.",
        "- The 500-window unified benchmark falls back to the full window artifact because only 209 random-test windows are available, creating potential training-data reuse.",
        "",
        "## Reproducibility",
        "",
        "All experiment entry points live under `scripts/` and write their outputs under `logs/`. Configuration values are maintained in `config.yaml`.",
        "",
    ])
    output = ROOT / "REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output}")
    return output


if __name__ == "__main__":
    generate_report()
