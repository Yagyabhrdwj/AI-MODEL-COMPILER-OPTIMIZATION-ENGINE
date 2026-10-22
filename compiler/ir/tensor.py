from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tensor:
    tensor_id: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    producer: str | None = None
    consumers: set[str] = field(default_factory=set)
    value: Any = None
    quantization_scale: float | None = None
    is_constant: bool = False
    is_graph_input: bool = False
    is_graph_output: bool = False

    def describe(self) -> str:
        flags = []
        if self.is_constant:
            flags.append("constant")
        if self.is_graph_input:
            flags.append("input")
        if self.is_graph_output:
            flags.append("output")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        return f"{self.tensor_id}: shape={self.shape}, dtype={self.dtype}{suffix}"
