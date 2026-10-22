from __future__ import annotations

import torch

from compiler.ir.graph import Graph


def eliminate_redundant_ops(graph: Graph) -> int:
    """Remove scalar x+0 and x*1 operations when broadcasting is unambiguous."""
    removed = 0
    for node in list(graph.topological_nodes()):
        if node.operation not in {"Add", "Mul"} or len(node.inputs) != 2:
            continue
        identity = 0 if node.operation == "Add" else 1
        constant_ids = [item for item in node.inputs if graph.tensors[item].is_constant]
        dynamic_ids = [item for item in node.inputs if not graph.tensors[item].is_constant]
        if len(constant_ids) != 1 or len(dynamic_ids) != 1:
            continue
        value = graph.tensors[constant_ids[0]].value
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or value.item() != identity:
            continue
        source_id = dynamic_ids[0]
        output_id = node.outputs[0]
        output_tensor = graph.tensors[output_id]
        for consumer_id in list(output_tensor.consumers):
            consumer = graph.nodes[consumer_id]
            consumer.inputs = [source_id if item == output_id else item for item in consumer.inputs]
            graph.tensors[source_id].consumers.add(consumer_id)
        if output_id in graph.outputs:
            graph.outputs = [source_id if item == output_id else item for item in graph.outputs]
            graph.tensors[source_id].is_graph_output = True
        graph.tensors[source_id].consumers.discard(node.node_id)
        graph.tensors.pop(output_id)
        graph.tensors[constant_ids[0]].consumers.discard(node.node_id)
        graph.nodes.pop(node.node_id)
        removed += 1
    graph.validate()
    return removed