from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from compiler.ir.graph import Graph


@dataclass(frozen=True)
class ActivationRange:
    minimum: float
    maximum: float
    scale: float


@dataclass(frozen=True)
class NumericalError:
    maximum_absolute: float
    mean_absolute: float
    maximum_relative: float


def calibrate_activations(
    graph: Graph,
    calibration_inputs: Iterable[dict[str, torch.Tensor]],
) -> dict[str, ActivationRange]:
    """Collect symmetric INT8 ranges for tensors observed during calibration."""
    from compiler.backend.reference import execute

    ranges: dict[str, list[float]] = {}
    batches = 0
    for inputs in calibration_inputs:
        batches += 1
        values = execute(graph, inputs, return_intermediates=True)
        for tensor_id, value in values.items():
            if graph.tensors[tensor_id].is_constant:
                continue
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            detached = value.detach()
            current = ranges.setdefault(tensor_id, [float("inf"), float("-inf")])
            current[0] = min(current[0], detached.min().item())
            current[1] = max(current[1], detached.max().item())
    if batches == 0:
        raise ValueError("calibration_inputs must contain at least one batch")

    return {
        tensor_id: ActivationRange(
            minimum=minimum,
            maximum=maximum,
            scale=max(abs(minimum), abs(maximum)) / 127.0 or 1.0,
        )
        for tensor_id, (minimum, maximum) in ranges.items()
    }


def numerical_error(reference: torch.Tensor, candidate: torch.Tensor) -> NumericalError:
    """Summarize absolute and relative error between two tensors."""
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate tensors must have the same shape")
    difference = (candidate.to(torch.float32) - reference.to(torch.float32)).abs()
    denominator = reference.to(torch.float32).abs().clamp_min(torch.finfo(torch.float32).eps)
    return NumericalError(
        maximum_absolute=difference.max().item(),
        mean_absolute=difference.mean().item(),
        maximum_relative=(difference / denominator).max().item(),
    )


def convert_constants(graph: Graph, dtype: torch.dtype) -> None:
    """Convert stored parameters to a reduced precision for reference execution."""
    for tensor in graph.tensors.values():
        if tensor.is_constant and isinstance(tensor.value, torch.Tensor):
            tensor.value = tensor.value.to(dtype)
            tensor.dtype = str(dtype).removeprefix("torch.")


def quantize_weights_int8(graph: Graph) -> int:
    """Quantize weight constants symmetrically and retain a dequantization scale."""
    converted = 0
    for tensor_id, tensor in graph.tensors.items():
        if not tensor.is_constant or not isinstance(tensor.value, torch.Tensor):
            continue
        live_consumers = tensor.consumers & graph.nodes.keys()
        tensor.consumers = live_consumers
        is_weight = any(
            graph.nodes[consumer_id].operation in {"Linear", "Conv2d", "FusedConvBatchNormReLU"}
            and graph.nodes[consumer_id].inputs.index(tensor_id) == 1
            for consumer_id in live_consumers
        )
        if not is_weight:
            continue
        maximum = tensor.value.detach().abs().max().item()
        scale = maximum / 127.0 if maximum else 1.0
        tensor.value = torch.clamp(torch.round(tensor.value / scale), -127, 127).to(torch.int8)
        tensor.quantization_scale = scale
        tensor.dtype = "int8"
        converted += 1
    graph.validate()
    return converted