import torch
from torch import nn

from compiler.backend import execute
from compiler.frontend import import_model


def test_import_and_reference_execution_are_equivalent():
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2)).eval()
    example = torch.randn(3, 4)

    graph = import_model(model, example)
    result = execute(graph, {graph.inputs[0]: example})[graph.outputs[0]]

    assert graph.inputs
    assert graph.outputs
    assert torch.allclose(result, model(example), atol=1e-6, rtol=1e-5)


def test_unsupported_module_has_a_clear_diagnostic():
    model = nn.Sequential(nn.Sigmoid()).eval()
    try:
        import_model(model, torch.randn(1, 2))
    except NotImplementedError as error:
        assert "Sigmoid" in str(error)
    else:
        raise AssertionError("Unsupported module was silently accepted")
