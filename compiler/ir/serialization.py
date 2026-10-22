from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .graph import Graph


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": value.detach().cpu().tolist(), "dtype": str(value.dtype).removeprefix("torch.")}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    return {
        "name": graph.name,
        "inputs": graph.inputs,
        "outputs": graph.outputs,
        "nodes": [
            {
                "node_id": node.node_id,
                "operation": node.operation,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "attributes": _json_value(node.attributes),
                "parameters": _json_value(node.parameters),
                "metadata": _json_value(node.metadata),
            }
            for node in graph.topological_nodes()
        ],
        "tensors": [
            {
                "tensor_id": tensor.tensor_id,
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "producer": tensor.producer,
                "value": _json_value(tensor.value) if tensor.is_constant else None,
                "quantization_scale": tensor.quantization_scale,
                "is_constant": tensor.is_constant,
                "is_graph_input": tensor.is_graph_input,
                "is_graph_output": tensor.is_graph_output,
            }
            for tensor in graph.tensors.values()
        ],
    }


def write_graph_json(graph: Graph, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(graph_to_dict(graph), indent=2), encoding="utf-8")