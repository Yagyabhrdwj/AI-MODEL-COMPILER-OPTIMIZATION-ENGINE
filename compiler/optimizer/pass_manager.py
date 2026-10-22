from __future__ import annotations

from collections.abc import Callable

from compiler.ir.graph import Graph


class PassManager:
    def __init__(self) -> None:
        self._passes: list[tuple[str, Callable[[Graph], int]]] = []

    def add(self, name: str, optimization: Callable[[Graph], int]) -> None:
        self._passes.append((name, optimization))

    def run(self, graph: Graph) -> dict[str, int]:
        statistics = {}
        for name, optimization in self._passes:
            statistics[name] = optimization(graph)
        graph.validate()
        return statistics
