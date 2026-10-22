from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_benchmark_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read persisted benchmark rows and convert numeric fields to numbers."""
    numeric_fields = {
        "runs", "warmup", "batch_size", "mean_ms", "median_ms", "p95_ms",
        "throughput_samples_per_second",
    }
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            for field in numeric_fields:
                if field in row:
                    row[field] = float(row[field]) if "." in row[field] else int(row[field])
            rows.append(row)
    return rows


def plot_benchmark_csv(input_path: str | Path, output_path: str | Path) -> None:
    """Write a latency and throughput plot from persisted benchmark results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate benchmark plots") from exc

    rows = read_benchmark_csv(input_path)
    if not rows:
        raise ValueError("benchmark CSV contains no rows")
    labels = [row["graph_name"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [row["mean_ms"] for row in rows], color="#2f6f8f")
    axes[0].set_title("Mean latency")
    axes[0].set_ylabel("Milliseconds")
    axes[1].bar(labels, [row["throughput_samples_per_second"] for row in rows], color="#d47b35")
    axes[1].set_title("Throughput")
    axes[1].set_ylabel("Samples / second")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)