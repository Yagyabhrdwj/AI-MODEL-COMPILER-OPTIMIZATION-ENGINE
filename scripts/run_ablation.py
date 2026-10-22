from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from benchmarks import append_result_csv, benchmark, write_result_json
from compiler.frontend import import_model
from compiler.optimizer import (
    PassManager,
    constant_fold,
    eliminate_dead_nodes,
    eliminate_redundant_ops,
    fuse_conv_batchnorm_relu,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    example = torch.randn(8, 16)
    configurations = {
        "baseline": [],
        "optimized": [
            ("constant_folding", constant_fold),
            ("dead_node_elimination", eliminate_dead_nodes),
            ("redundant_operation_elimination", eliminate_redundant_ops),
            ("conv_batchnorm_relu_fusion", fuse_conv_batchnorm_relu),
        ],
    }
    csv_path = RESULTS / "ablation.csv"
    if csv_path.exists():
        csv_path.unlink()
    for name, passes in configurations.items():
        graph = import_model(model, example)
        manager = PassManager()
        for pass_name, optimization in passes:
            manager.add(pass_name, optimization)
        stats = manager.run(graph)
        graph.name = f"mlp_demo_{name}"
        result = benchmark(graph, {graph.inputs[0]: example}, warmup=10, runs=50)
        result_path = RESULTS / f"{name}.json"
        write_result_json(result, result_path)
        append_result_csv(result, csv_path)
        print(f"{name}: passes={stats}, mean_ms={result.mean_ms:.4f}, throughput={result.throughput_samples_per_second:.2f}")


if __name__ == "__main__":
    main()