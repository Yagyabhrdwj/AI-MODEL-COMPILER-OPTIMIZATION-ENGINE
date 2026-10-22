from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from compiler.backend import execute
from compiler.ir.graph import Graph


@dataclass(frozen=True)
class BenchmarkResult:
    graph_name: str
    runs: int
    warmup: int
    batch_size: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_samples_per_second: float
    python_version: str
    torch_version: str
    device: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark(
    graph: Graph,
    inputs: dict[str, torch.Tensor],
    *,
    warmup: int = 10,
    runs: int = 50,
) -> BenchmarkResult:
    if warmup < 0 or runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")
    nodes = graph.topological_nodes()
    for _ in range(warmup):
        execute(graph, inputs, nodes=nodes)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        execute(graph, inputs, nodes=nodes)
        timings.append((time.perf_counter() - start) * 1000)
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    mean_ms = statistics.fmean(timings)
    batch_size = next(iter(inputs.values())).shape[0]
    return BenchmarkResult(
        graph_name=graph.name,
        runs=runs,
        warmup=warmup,
        batch_size=batch_size,
        mean_ms=mean_ms,
        median_ms=statistics.median(timings),
        p95_ms=p95,
        throughput_samples_per_second=batch_size / (mean_ms / 1000),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        device=str(next(iter(inputs.values())).device),
    )
