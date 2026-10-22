from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .runner import BenchmarkResult


def write_result_json(result: BenchmarkResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def append_result_csv(result: BenchmarkResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    write_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)