from __future__ import annotations

from typing import Any

import torch
from torch import nn

from compiler.backend import execute
from compiler.frontend import import_model
from compiler.quantization import numerical_error

from .runner import benchmark


def evaluate_model(
    name: str,
    model: nn.Module,
    example_input: torch.Tensor,
    *,
    warmup: int = 2,
    runs: int = 5,
) -> dict[str, Any]:
    """Import, check, and benchmark one model without hiding unsupported operators."""
    model = model.eval()
    try:
        graph = import_model(model, example_input)
        compiled = execute(graph, {graph.inputs[0]: example_input})[graph.outputs[0]]
        reference = model(example_input)
        error = numerical_error(reference, compiled)
        result = benchmark(graph, {graph.inputs[0]: example_input}, warmup=warmup, runs=runs)
    except (NotImplementedError, ValueError, RuntimeError) as exc:
        return {"name": name, "status": "skipped", "reason": str(exc)}

    return {
        "name": name,
        "status": "success",
        "nodes": len(graph.nodes),
        "tensors": len(graph.tensors),
        "maximum_absolute_error": error.maximum_absolute,
        "mean_absolute_error": error.mean_absolute,
        "maximum_relative_error": error.maximum_relative,
        "benchmark": result.to_dict(),
    }