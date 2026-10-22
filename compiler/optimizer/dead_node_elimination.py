from __future__ import annotations

from compiler.ir.graph import Graph


def eliminate_dead_nodes(graph: Graph) -> int:
    live = graph.reachable_nodes()
    dead = [node_id for node_id in graph.nodes if node_id not in live]
    for node_id in dead:
        graph.remove_node(node_id)
    graph.validate()
    return len(dead)
