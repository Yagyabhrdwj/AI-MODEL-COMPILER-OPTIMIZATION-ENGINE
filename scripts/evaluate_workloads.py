from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch
from torch import nn

from benchmarks import evaluate_model


def _torchvision_workloads() -> dict[str, Callable[[], nn.Module]]:
    try:
        from torchvision import models
    except ImportError:
        return {}
    return {
        "resnet18": lambda: models.resnet18(weights=None),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate controlled CNN workloads through the custom IR")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    workloads = _torchvision_workloads()
    if not workloads:
        print(json.dumps({
            "status": "unavailable",
            "reason": "torchvision is not installed; install a compatible torchvision wheel to run CNN workloads",
            "workloads": [],
        }, indent=2))
        return

    example = torch.randn(1, 3, 224, 224)
    results = [
        evaluate_model(name, factory(), example, warmup=args.warmup, runs=args.runs)
        for name, factory in workloads.items()
    ]
    print(json.dumps({"status": "complete", "workloads": results}, indent=2))


if __name__ == "__main__":
    main()