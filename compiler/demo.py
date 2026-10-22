from __future__ import annotations

import torch
from torch import nn

from compiler.backend import execute
from compiler.frontend import import_model
from compiler.optimizer import eliminate_dead_nodes


def main() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    example = torch.randn(1, 4)
    graph = import_model(model, example)
    removed = eliminate_dead_nodes(graph)
    compiled_output = execute(graph, {graph.inputs[0]: example})[graph.outputs[0]]
    reference_output = model(example)
    error = (compiled_output - reference_output).abs().max().item()
    print(f"Imported {len(graph.nodes)} nodes; removed {removed} dead nodes")
    print(f"Maximum absolute output error: {error:.8f}")


if __name__ == "__main__":
    main()
