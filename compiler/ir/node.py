from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    node_id: str
    operation: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
