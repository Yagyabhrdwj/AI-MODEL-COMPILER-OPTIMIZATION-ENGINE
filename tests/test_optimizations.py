import torch

from compiler.backend import execute
from compiler.ir import Graph, Node, Tensor
from compiler.optimizer import constant_fold, eliminate_dead_nodes, eliminate_redundant_ops


def test_constant_folding_replaces_constant_add():
    graph = Graph(name="constants")
    graph.add_tensor(Tensor("left", shape=(), dtype="float32", value=torch.tensor(2.0), is_constant=True))
    graph.add_tensor(Tensor("right", shape=(), dtype="float32", value=torch.tensor(3.0), is_constant=True))
    graph.add_node(Node("add", "Add", ["left", "right"], ["sum"]))
    graph.outputs.append("sum")
    graph.tensors["sum"].is_graph_output = True

    assert constant_fold(graph) == 1
    assert not graph.nodes
    assert graph.tensors["sum"].is_constant
    assert torch.equal(execute(graph, {})["sum"], torch.tensor(5.0))


def test_dead_node_elimination_preserves_requested_output():
    graph = Graph(name="dead")
    graph.add_tensor(Tensor("input", shape=(1,), dtype="float32", is_graph_input=True))
    graph.inputs.append("input")
    graph.add_node(Node("live", "ReLU", ["input"], ["live_out"]))
    graph.add_node(Node("dead", "ReLU", ["input"], ["dead_out"]))
    graph.outputs.append("live_out")
    graph.tensors["live_out"].is_graph_output = True
    graph.validate()

    assert eliminate_dead_nodes(graph) == 1
    assert set(graph.nodes) == {"live"}
    assert graph.outputs == ["live_out"]


def test_redundant_scalar_addition_is_removed():
    graph = Graph(name="redundant")
    graph.add_tensor(Tensor("input", shape=(2,), dtype="float32", is_graph_input=True))
    graph.add_tensor(Tensor("zero", shape=(), dtype="float32", value=torch.tensor(0.0), is_constant=True))
    graph.inputs.append("input")
    graph.add_node(Node("add", "Add", ["input", "zero"], ["output"]))
    graph.outputs.append("output")
    graph.tensors["output"].is_graph_output = True

    assert eliminate_redundant_ops(graph) == 1
    assert graph.outputs == ["input"]
    assert not graph.nodes
