from compiler.ir.graph import Graph


def to_dot(graph: Graph) -> str:
    """Return a Graphviz DOT representation without requiring Graphviz."""
    lines = [f'digraph "{graph.name}" {{', '  rankdir=LR;']
    for tensor_id, tensor in graph.tensors.items():
        if tensor.producer is None and not tensor.is_constant:
            lines.append(f'  "tensor:{tensor_id}" [shape=oval, label="{tensor_id}"];')
    for node in graph.topological_nodes():
        label = f"{node.node_id}\\n{node.operation}"
        lines.append(f'  "node:{node.node_id}" [shape=box, label="{label}"];')
        for tensor_id in node.inputs:
            lines.append(f'  "tensor:{tensor_id}" -> "node:{node.node_id}";')
        for tensor_id in node.outputs:
            lines.append(f'  "node:{node.node_id}" -> "tensor:{tensor_id}";')
    for tensor_id in graph.outputs:
        lines.append(f'  "tensor:{tensor_id}" [shape=oval, label="{tensor_id}"];')
    lines.append("}")
    return "\n".join(lines)
