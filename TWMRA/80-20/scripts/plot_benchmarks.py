"""Generate benchmark figures from the latest model comparison log."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_latest_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine benchmark rows with the latest optimized neural metrics."""
    benchmark = pd.read_csv(ROOT / "logs/benchmark.csv")
    optimized_rows = []
    for name in ("tcnca", "tst"):
        metrics_path = ROOT / f"logs/{name}_optimized_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        optimized_rows.append(
            {
                "model": f"{name}_optimized",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mae": metrics["mae"],
                "latency_ms": float("nan"),
                "num_params": metrics["num_params"],
            }
        )

    latest = benchmark.copy()
    if optimized_rows:
        latest = pd.concat([latest, pd.DataFrame(optimized_rows)], ignore_index=True)
    latest_path = ROOT / "logs/latest_model_comparison.csv"
    latest.to_csv(latest_path, index=False)
    return benchmark, latest


def save_bar_plot(data: pd.DataFrame, column: str, title: str, ylabel: str, output: Path) -> None:
    """Save one labeled bar chart for a benchmark metric."""
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(data["model"], data[column], color="#2f6f8f")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar in bars:
        axis.annotate(
            f"{bar.get_height():.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_benchmarks() -> list[Path]:
    """Read saved metrics and write refreshed benchmark and optimized figures."""
    benchmark_path = ROOT / "logs/benchmark.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark log not found: {benchmark_path}")

    benchmark, data = load_latest_comparison()
    output_dir = ROOT / "logs/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for column, title, ylabel, filename in [
        ("accuracy", "Model Accuracy", "Accuracy", "accuracy.png"),
        ("macro_f1", "Model Macro F1", "Macro F1", "macro_f1.png"),
        ("mae", "Encoded-Label Mean Absolute Error", "MAE", "mae.png"),
    ]:
        output = output_dir / filename
        plot_data = data.dropna(subset=[column])
        save_bar_plot(plot_data, column, title, ylabel, output)
        figures.append(output)

    latency_output = output_dir / "latency_ms.png"
    save_bar_plot(benchmark, "latency_ms", "Measured Deployment Inference Latency", "Median latency (ms)", latency_output)
    figures.append(latency_output)

    optimized = data[data["model"].str.endswith("_optimized")].copy()
    if not optimized.empty:
        optimized_output = output_dir / "optimized_neural_comparison.png"
        figure, axis = plt.subplots(figsize=(9, 5))
        x = range(len(optimized))
        width = 0.35
        axis.bar([value - width / 2 for value in x], optimized["accuracy"], width, label="Accuracy")
        axis.bar([value + width / 2 for value in x], optimized["macro_f1"], width, label="Macro F1")
        axis.set_xticks(list(x), optimized["model"])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_title("Latest Optimized Neural Models")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(optimized_output, dpi=160)
        plt.close(figure)
        figures.append(optimized_output)

    print(data.to_string(index=False))
    print(f"Saved comparison table: {ROOT / 'logs/latest_model_comparison.csv'}")
    for figure in figures:
        print(f"Saved figure: {figure}")
    return figures


if __name__ == "__main__":
    plot_benchmarks()
