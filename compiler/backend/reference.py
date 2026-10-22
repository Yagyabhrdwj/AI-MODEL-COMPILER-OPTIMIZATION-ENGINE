from __future__ import annotations

import operator
from collections.abc import Iterable

import torch
import torch.nn.functional as functional

from compiler.ir.graph import Graph
from compiler.ir.node import Node


def execute(
    graph: Graph,
    inputs: dict[str, torch.Tensor],
    *,
    return_intermediates: bool = False,
    nodes: Iterable[Node] | None = None,
) -> dict[str, torch.Tensor]:
    values = {**inputs}
    values.update({
        tensor_id: (
            tensor.value.to(torch.float32) * tensor.quantization_scale
            if tensor.quantization_scale is not None else tensor.value
        )
        for tensor_id, tensor in graph.tensors.items() if tensor.is_constant
    })
    for node in graph.topological_nodes() if nodes is None else nodes:
        args = [values[tensor_id] for tensor_id in node.inputs]
        if node.operation == "Linear":
            result = functional.linear(args[0], args[1], args[2] if len(args) > 2 else None)
        elif node.operation == "Conv2d":
            result = functional.conv2d(
                args[0], args[1], args[2] if len(args) > 2 else None,
                stride=node.parameters["stride"], padding=node.parameters["padding"],
                dilation=node.parameters["dilation"], groups=node.parameters["groups"],
            )
        elif node.operation == "FusedConvBatchNormReLU":
            result = functional.relu(functional.conv2d(
                args[0], args[1], args[2], stride=node.parameters["stride"],
                padding=node.parameters["padding"], dilation=node.parameters["dilation"],
                groups=node.parameters["groups"],
            ))
        elif node.operation == "BatchNorm2d":
            result = functional.batch_norm(
                args[0], args[3], args[4], args[1], args[2], training=False,
                momentum=node.parameters["momentum"], eps=node.parameters["eps"],
            )
        elif node.operation in {"MaxPool2d", "AvgPool2d"}:
            pool = functional.max_pool2d if node.operation == "MaxPool2d" else functional.avg_pool2d
            result = pool(
                args[0], kernel_size=node.parameters["kernel_size"], stride=node.parameters["stride"],
                padding=node.parameters["padding"], dilation=node.parameters["dilation"],
                ceil_mode=node.parameters["ceil_mode"],
            )
        elif node.operation == "Flatten":
            result = torch.flatten(args[0], start_dim=node.parameters["start_dim"], end_dim=node.parameters["end_dim"])
        elif node.operation == "ReLU":
            result = functional.relu(args[0])
        elif node.operation == "Add":
            result = operator.add(*args)
        elif node.operation == "Mul":
            result = operator.mul(*args)
        elif node.operation == "Constant":
            result = node.attributes["value"]
        else:
            raise NotImplementedError(f"Reference backend does not support {node.operation}")
        for output_id in node.outputs:
            values[output_id] = result
    if return_intermediates:
        return values
    return {tensor_id: values[tensor_id] for tensor_id in graph.outputs}
