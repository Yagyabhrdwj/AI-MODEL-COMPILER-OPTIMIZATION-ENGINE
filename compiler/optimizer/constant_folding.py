from __future__ import annotations

import operator

from compiler.ir.graph import Graph
from compiler.ir.node import Node
from compiler.ir.tensor import Tensor


def constant_fold(graph: Graph) -> int:
    folded = 0
    for node in list(graph.topological_nodes()):
        if node.operation not in {"Add", "Mul"} or not all(graph.tensors[item].is_constant for item in node.inputs):
            continue
        values = [graph.tensors[item].value for item in node.inputs]
        result = operator.add(*values) if node.operation == "Add" else operator.mul(*values)
        output_id = node.outputs[0]
        output = graph.tensors[output_id]
        output.value = result
        output.is_constant = True
        output.producer = None
        output.consumers = set(output.consumers)
        for input_id in node.inputs:
            graph.tensors[input_id].consumers.discard(node.node_id)
        graph.nodes.pop(node.node_id)
        folded += 1
    graph.validate()
    return folded
