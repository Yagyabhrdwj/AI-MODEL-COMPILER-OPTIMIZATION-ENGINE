import torch

from compiler.gui import compile_workload


def test_gui_compile_pipeline_returns_equivalent_mlp_result():
    result = compile_workload("MLP demo")

    assert len(result.graph.nodes) == 3
    assert torch.allclose(result.compiled_output, result.reference_output, atol=1e-6, rtol=1e-5)


def test_gui_compile_pipeline_supports_cnn_and_int8():
    result = compile_workload("CNN demo", precision="INT8")

    assert "Conv2d" in {node.operation for node in result.graph.topological_nodes()} or "FusedConvBatchNormReLU" in {node.operation for node in result.graph.topological_nodes()}
    assert torch.isfinite(result.compiled_output).all()