from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from .node import Node
from .tensor import Tensor


@dataclass
class Graph:
    name: str = "graph"
    nodes: dict[str, Node] = field(default_factory=dict)
    tensors: dict[str, Tensor] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def add_tensor(self, tensor: Tensor) -> None:
        if tensor.tensor_id in self.tensors:
            raise ValueError(f"Tensor already exists: {tensor.tensor_id}")
        self.tensors[tensor.tensor_id] = tensor

    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node already exists: {node.node_id}")
        missing = [tensor_id for tensor_id in node.inputs if tensor_id not in self.tensors]
        if missing:
            raise ValueError(f"Node {node.node_id} references missing tensors: {missing}")
        for tensor_id in node.outputs:
            if tensor_id in self.tensors:
                raise ValueError(f"Output tensor already exists: {tensor_id}")
            self.add_tensor(Tensor(tensor_id=tensor_id, producer=node.node_id))
        self.nodes[node.node_id] = node
        for tensor_id in node.inputs:
            self.tensors[tensor_id].consumers.add(node.node_id)

    def topological_nodes(self) -> list[Node]:
        pending = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            pending[node.node_id] = sum(
                1 for tensor_id in node.inputs
                if self.tensors[tensor_id].producer is not None
            )
        ready = deque(node_id for node_id, count in pending.items() if count == 0)
        ordered = []
        while ready:
            node_id = ready.popleft()
            ordered.append(self.nodes[node_id])
            for output_id in self.nodes[node_id].outputs:
                for consumer_id in self.tensors[output_id].consumers:
                    pending[consumer_id] -= 1
                    if pending[consumer_id] == 0:
                        ready.append(consumer_id)
        if len(ordered) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return ordered

    def remove_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id)
        for tensor_id in node.inputs:
            self.tensors[tensor_id].consumers.discard(node_id)
        for tensor_id in node.outputs:
            tensor = self.tensors.pop(tensor_id)
            for consumer_id in tensor.consumers:
                self.nodes[consumer_id].inputs = [
                    item for item in self.nodes[consumer_id].inputs if item != tensor_id
                ]

    def validate(self) -> None:
        if not self.outputs:
            raise ValueError("Graph must declare at least one output")
        for tensor_id in self.inputs + self.outputs:
            if tensor_id not in self.tensors:
                raise ValueError(f"Graph references missing boundary tensor: {tensor_id}")
        for node in self.nodes.values():
            for tensor_id in node.inputs:
                if tensor_id not in self.tensors:
                    raise ValueError(f"Node {node.node_id} references missing input {tensor_id}")
                if node.node_id not in self.tensors[tensor_id].consumers:
                    raise ValueError(f"Consumer link missing for {tensor_id}")
            for tensor_id in node.outputs:
                if tensor_id not in self.tensors:
                    raise ValueError(f"Node {node.node_id} references missing output {tensor_id}")
                if self.tensors[tensor_id].producer != node.node_id:
                    raise ValueError(f"Producer link invalid for {tensor_id}")
        self.topological_nodes()

    def reachable_nodes(self) -> set[str]:
        reachable: set[str] = set()
        stack = [self.tensors[tensor_id].producer for tensor_id in self.outputs]
        while stack:
            node_id = stack.pop()
            if node_id is None or node_id in reachable:
                continue
            reachable.add(node_id)
            stack.extend(
                self.tensors[tensor_id].producer
                for tensor_id in self.nodes[node_id].inputs
            )
        return reachable
