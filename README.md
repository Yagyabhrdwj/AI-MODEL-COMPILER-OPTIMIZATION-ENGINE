# AI Model Compiler & Optimization Engine

A research-oriented ML compiler prototype for importing selected PyTorch models into a custom intermediate representation (IR), validating and optimizing the graph, and comparing reference execution results.

## Current vertical slice

The first milestone supports:

- `torch.fx` import for `Linear`, `ReLU`, `Conv2d`, `BatchNorm2d`, pooling, `Flatten`, and selected elementwise operators
- Custom graph IR with nodes, tensors, shapes, dtypes, producers, and consumers
- Structural validation and cycle detection
- Constant folding for scalar/tensor `add` and `mul` constants
- Dead-node elimination
- A PyTorch reference backend for the imported IR
- DOT graph visualization without a Graphviz runtime dependency
- Repeated latency benchmarking with environment metadata
- End-to-end numerical equivalence tests
- JSON-safe IR serialization and CSV/JSON benchmark result persistence
- Baseline-versus-optimized ablation runner
- Symmetric weight-only INT8 quantization with reference dequantization
- Windows desktop GUI workbench for compiling, benchmarking, and exporting demo workloads

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch is currently pinned to a CPU-friendly minimum in `requirements.txt`; use the appropriate wheel for your platform if needed.

## Run the vertical slice

```powershell
python -m compiler.demo
python -m scripts.benchmark_demo
python -m scripts.run_ablation
python -m scripts.evaluate_workloads
python -m scripts.generate_benchmark_plot
python -m compiler.gui
pytest
.venv\Scripts\python.exe docs\generate_status_pdf.py
```

The benchmark command reports mean, median, and p95 latency plus throughput. It is a measurement tool, not a claim of performance improvement.

The ablation command writes baseline and optimized measurements to `results/ablation.csv` and individual JSON files. Results are workload- and hardware-specific; they must be interpreted experimentally.

The workload command evaluates torchvision `resnet18` and `mobilenet_v3_small` when a compatible torchvision installation is available. Unsupported operators are reported as skipped with their diagnostic; no model weights or network access are required.

The plot command reads `results/ablation.csv` and writes `results/ablation.png`; install matplotlib separately when plot generation is needed.

## Project blueprint

See [docs/PROJECT_BLUEPRINT.md](docs/PROJECT_BLUEPRINT.md) for the target architecture, IR contract, milestones, risks, and research methodology.

The current implementation report is available at [docs/AI_MODEL_COMPILER_STATUS.pdf](docs/AI_MODEL_COMPILER_STATUS.pdf). Regenerate it with `docs/generate_status_pdf.py` after significant code updates.

## Scope discipline

This repository intentionally starts with a small, testable operator subset. CNN import, fusion, quantization, benchmarking, and visualization are planned milestones, not claims about the current implementation.
