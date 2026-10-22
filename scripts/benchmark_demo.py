from __future__ import annotations

import json

import torch
from torch import nn

from benchmarks import benchmark
from compiler.frontend import import_model


def main() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    example = torch.randn(8, 16)
    graph = import_model(model, example)
    result = benchmark(graph, {graph.inputs[0]: example}, warmup=3, runs=10)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
