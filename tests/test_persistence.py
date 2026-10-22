import json

import torch
from torch import nn

from benchmarks import append_result_csv, benchmark, write_result_json
from compiler.frontend import import_model
from compiler.ir import graph_to_dict, write_graph_json
from visualization.benchmarks import read_benchmark_csv


def test_graph_serialization_is_json_safe(tmp_path):
    model = nn.Linear(2, 1).eval()
    example = torch.ones(1, 2)
    graph = import_model(model, example)

    payload = graph_to_dict(graph)
    output = tmp_path / "graph.json"
    write_graph_json(graph, output)

    assert payload["nodes"]
    assert json.loads(output.read_text(encoding="utf-8"))["name"] == "Linear"


def test_benchmark_results_can_be_persisted(tmp_path):
    model = nn.Linear(2, 1).eval()
    example = torch.ones(2, 2)
    graph = import_model(model, example)
    result = benchmark(graph, {graph.inputs[0]: example}, warmup=0, runs=1)
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "results.csv"

    write_result_json(result, json_path)
    append_result_csv(result, csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["runs"] == 1
    assert "mean_ms" in csv_path.read_text(encoding="utf-8").splitlines()[0]


def test_benchmark_csv_reader_converts_numeric_fields(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text(
        "graph_name,runs,mean_ms,throughput_samples_per_second\n"
        "demo,2,1.5,4.0\n",
        encoding="utf-8",
    )

    row = read_benchmark_csv(path)[0]

    assert row["runs"] == 2
    assert row["mean_ms"] == 1.5
    assert row["throughput_samples_per_second"] == 4.0