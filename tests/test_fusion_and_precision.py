import torch
import pytest
from torch import nn

from compiler.backend import execute
from compiler.frontend import import_model
from compiler.optimizer import fuse_conv_batchnorm_relu
from compiler.quantization import (
    calibrate_activations,
    convert_constants,
    numerical_error,
    quantize_weights_int8,
)


def test_conv_batchnorm_relu_fusion_preserves_output():
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
    ).eval()
    example = torch.randn(2, 3, 8, 8)
    graph = import_model(model, example)
    expected = execute(graph, {graph.inputs[0]: example})[graph.outputs[0]]

    assert fuse_conv_batchnorm_relu(graph) == 1
    actual = execute(graph, {graph.inputs[0]: example})[graph.outputs[0]]

    assert [node.operation for node in graph.topological_nodes()] == ["FusedConvBatchNormReLU"]
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_fp16_constant_conversion_executes():
    torch.manual_seed(8)
    model = nn.Linear(3, 2).eval().half()
    example = torch.randn(4, 3).half()
    graph = import_model(model, example)
    convert_constants(graph, torch.float16)

    result = execute(graph, {graph.inputs[0]: example})[graph.outputs[0]]

    assert result.dtype == torch.float16
    assert all(
        tensor.dtype == "float16"
        for tensor in graph.tensors.values()
        if tensor.is_constant and isinstance(tensor.value, torch.Tensor)
    )


def test_int8_weight_quantization_preserves_reasonable_output():
    torch.manual_seed(12)
    model = nn.Linear(4, 2).eval()
    example = torch.randn(3, 4)
    graph = import_model(model, example)
    expected = execute(graph, {graph.inputs[0]: example})[graph.outputs[0]]

    assert quantize_weights_int8(graph) == 1
    actual = execute(graph, {graph.inputs[0]: example})[graph.outputs[0]]

    weight_tensor = next(
        tensor for tensor in graph.tensors.values()
        if tensor.quantization_scale is not None
    )
    assert weight_tensor.dtype == "int8"
    assert torch.allclose(actual, expected, atol=0.02, rtol=0.02)


def test_activation_calibration_collects_ranges_and_scales():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU()).eval()
    graph = import_model(model, torch.ones(1, 2))

    ranges = calibrate_activations(
        graph,
        [
            {graph.inputs[0]: torch.tensor([[1.0, -2.0]])},
            {graph.inputs[0]: torch.tensor([[3.0, 4.0]])},
        ],
    )

    output_range = ranges[graph.outputs[0]]
    assert output_range.minimum >= 0.0
    assert output_range.maximum >= output_range.minimum
    assert output_range.scale > 0.0


def test_numerical_error_reports_absolute_and_relative_metrics():
    report = numerical_error(torch.tensor([1.0, 2.0]), torch.tensor([1.1, 1.0]))

    assert report.maximum_absolute == pytest.approx(1.0)
    assert report.mean_absolute == pytest.approx(0.55)
    assert report.maximum_relative == pytest.approx(0.5)
