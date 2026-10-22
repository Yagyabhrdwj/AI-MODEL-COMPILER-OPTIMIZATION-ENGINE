import torch
from torch import nn

from benchmarks import benchmark, evaluate_model
from compiler.frontend import import_model
from visualization import to_dot


def test_dot_visualization_contains_nodes_and_edges():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU()).eval()
    example = torch.ones(1, 2)
    graph = import_model(model, example)

    dot = to_dot(graph)

    assert "Linear" in dot
    assert "ReLU" in dot
    assert "->" in dot


def test_benchmark_records_reproducible_metadata():
    model = nn.Linear(2, 1).eval()
    example = torch.ones(4, 2)
    graph = import_model(model, example)

    result = benchmark(graph, {graph.inputs[0]: example}, warmup=1, runs=3)

    assert result.runs == 3
    assert result.batch_size == 4
    assert result.mean_ms > 0
    assert result.throughput_samples_per_second > 0
    assert result.torch_version


def test_workload_evaluation_reports_accuracy_and_benchmark():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU()).eval()
    result = evaluate_model("test_mlp", model, torch.ones(4, 2), warmup=0, runs=1)

    assert result["status"] == "success"
    assert result["maximum_absolute_error"] == 0.0
    assert result["benchmark"]["runs"] == 1


def test_workload_evaluation_reports_unsupported_models():
    result = evaluate_model("unsupported", nn.Sigmoid(), torch.ones(1, 2), warmup=0, runs=1)

    assert result["status"] == "skipped"
    assert "Unsupported" in result["reason"]
