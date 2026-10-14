"""Summarize separate model-tuning results and generate final comparison plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TUNING_FILES = {
    "xgboost": "logs/xgboost_tuning.csv",
    "tcnca": "logs/tcnca_tuning.csv",
    "tst": "logs/tst_tuning.csv",
    "resnet1d": "logs/resnet1d_tuning.csv",
    "gru": "logs/gru_tuning.csv",
    "s5": "logs/s5_tuning.csv",
}


def summarize_tuning() -> pd.DataFrame:
    """Select the best macro-F1 candidate from every completed tuning run."""
    rows = []
    for model, relative_path in TUNING_FILES.items():
        path = ROOT / relative_path
        if not path.exists():
            continue
        data = pd.read_csv(path)
        best = data.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0].to_dict()
        best["model"] = model
        best["within_10ms"] = bool(best["latency_ms"] <= 10.0)
        rows.append(best)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise FileNotFoundError("No tuning CSV files were found.")
    benchmark_path = ROOT / "logs/benchmark.csv"
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)[["model", "latency_ms"]].rename(
            columns={"latency_ms": "benchmark_latency_ms"}
        )
        summary = summary.merge(benchmark, on="model", how="left")
    summary = summary.rename(columns={"latency_ms": "tuning_latency_ms"})
    summary["within_10ms"] = summary["benchmark_latency_ms"].le(10.0)
    summary = summary.sort_values(["macro_f1", "accuracy"], ascending=False)
    output = ROOT / "logs/tuning_summary.csv"
    summary.to_csv(output, index=False)
    (ROOT / "logs/tuning_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(summary["model"], summary["accuracy"], color="#2f6f8f")
    axes[0].set_title("Best tuned configuration: accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(summary["model"], summary["mae"], color="#c44e52")
    axes[1].set_title("Best tuned configuration: encoded-label MAE")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure_path = ROOT / "logs/figures/tuning/tuning_summary.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(summary[["model", "accuracy", "macro_f1", "mae", "tuning_latency_ms", "benchmark_latency_ms", "num_params", "within_10ms"]].to_string(index=False))
    print(f"Saved summary: {output}")
    print(f"Saved summary plot: {figure_path}")
    return summary


if __name__ == "__main__":
    summarize_tuning()
