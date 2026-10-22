from __future__ import annotations

import json
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

# Project Root
ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

PORT = 8080


class CompilerWebHandler(SimpleHTTPRequestHandler):
    """Zero-dependency HTTP handler serving the UI and providing compiler APIs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._send_json({
                "status": "online",
                "engine": "AI Model Compiler & Optimization Engine",
                "version": "0.1.0",
                "python_version": sys.version,
                "project_root": str(ROOT_DIR),
            })
            return
        super().do_GET()

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}

        if self.path == "/api/compile":
            self._handle_compile(payload)
            return

        if self.path == "/api/benchmark":
            self._handle_benchmark(payload)
            return

        self.send_error(404, "Endpoint not found")

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_compile(self, payload: dict[str, Any]) -> None:
        workload = payload.get("workload", "cnn_demo")
        precision = payload.get("precision", "FP32")
        passes = payload.get("passes", {})

        try:
            import torch
            from torch import nn
            from compiler.frontend import import_model
            from compiler.ir import graph_to_dict
            from compiler.optimizer import (
                PassManager,
                constant_fold,
                eliminate_dead_nodes,
                eliminate_redundant_ops,
                fuse_conv_batchnorm_relu,
            )
            from compiler.quantization import convert_constants, quantize_weights_int8

            # Build workload
            torch.manual_seed(0)
            if workload == "cnn_demo":
                model = nn.Sequential(
                    nn.Conv2d(3, 4, kernel_size=3, padding=1),
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(4 * 4 * 4, 3),
                ).eval()
                example_input = torch.randn(payload.get("batch_size", 8), 3, 8, 8)
            else:
                model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
                example_input = torch.randn(payload.get("batch_size", 8), 16)

            if precision == "FP16":
                model = model.half()
                example_input = example_input.half()

            graph = import_model(model, example_input)

            # Run passes
            manager = PassManager()
            if passes.get("constantFolding", True):
                manager.add("constant_folding", constant_fold)
            if passes.get("deadNodeElimination", True):
                manager.add("dead_node_elimination", eliminate_dead_nodes)
            if passes.get("redundantOps", True):
                manager.add("redundant_operation_elimination", eliminate_redundant_ops)
            if passes.get("fusion", True):
                manager.add("conv_batchnorm_relu_fusion", fuse_conv_batchnorm_relu)

            stats = manager.run(graph)

            if precision == "INT8":
                quantize_weights_int8(graph)
            elif precision == "FP16":
                convert_constants(graph, torch.float16)

            self._send_json({
                "status": "success",
                "stats": stats,
                "graph": graph_to_dict(graph),
            })
        except Exception as exc:
            self._send_json({"status": "fallback", "error": str(exc)}, status=200)

    def _handle_benchmark(self, payload: dict[str, Any]) -> None:
        try:
            import torch
            from benchmarks import benchmark
            from compiler.gui import make_workload
            from compiler.frontend import import_model

            workload = payload.get("workload", "cnn_demo")
            model, example = make_workload("CNN demo" if workload == "cnn_demo" else "MLP demo")
            graph = import_model(model, example)

            res = benchmark(
                graph,
                {graph.inputs[0]: example},
                warmup=payload.get("warmup", 5),
                runs=payload.get("runs", 30),
            )
            self._send_json({"status": "success", "result": res.to_dict()})
        except Exception as exc:
            self._send_json({"status": "fallback", "error": str(exc)}, status=200)


def main() -> None:
    print(f"================================================================")
    print(f"  AI Model Compiler & Optimization Engine | Web Dashboard")
    print(f"================================================================")
    print(f"  Local URL:  http://localhost:{PORT}")
    print(f"  Frontend:   {FRONTEND_DIR}")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"================================================================")

    webbrowser.open(f"http://localhost:{PORT}")
    server = HTTPServer(("0.0.0.0", PORT), CompilerWebHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.server_close()


if __name__ == "__main__":
    main()
