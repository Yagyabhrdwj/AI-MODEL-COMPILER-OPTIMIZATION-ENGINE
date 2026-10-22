from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import torch
from torch import nn

from benchmarks import benchmark
from compiler.backend import execute
from compiler.frontend import import_model
from compiler.ir import Graph, graph_to_dict, write_graph_json
from compiler.optimizer import (
    PassManager,
    constant_fold,
    eliminate_dead_nodes,
    eliminate_redundant_ops,
    fuse_conv_batchnorm_relu,
)
from compiler.quantization import convert_constants, quantize_weights_int8


@dataclass
class CompilationResult:
    graph: Graph
    example_input: torch.Tensor
    reference_output: torch.Tensor
    compiled_output: torch.Tensor
    pass_stats: dict[str, int]
    precision: str


def make_workload(name: str) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    if name == "CNN demo":
        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 3),
        ).eval()
        return model, torch.randn(2, 3, 8, 8)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    return model, torch.randn(8, 16)


def compile_workload(
    workload: str,
    precision: str = "FP32",
    enabled_passes: tuple[str, ...] = (
        "Constant folding",
        "Dead-node elimination",
        "Redundant operation elimination",
        "Conv-BN-ReLU fusion",
    ),
) -> CompilationResult:
    model, example_input = make_workload(workload)
    if precision == "FP16":
        model = model.half()
        example_input = example_input.half()
    graph = import_model(model, example_input)
    reference_output = model(example_input)
    manager = PassManager()
    pass_map = {
        "Constant folding": constant_fold,
        "Dead-node elimination": eliminate_dead_nodes,
        "Redundant operation elimination": eliminate_redundant_ops,
        "Conv-BN-ReLU fusion": fuse_conv_batchnorm_relu,
    }
    for pass_name in enabled_passes:
        manager.add(pass_name, pass_map[pass_name])
    pass_stats = manager.run(graph)
    if precision == "INT8":
        quantize_weights_int8(graph)
    elif precision == "FP16":
        convert_constants(graph, torch.float16)
    compiled_output = execute(graph, {graph.inputs[0]: example_input})[graph.outputs[0]]
    return CompilationResult(graph, example_input, reference_output, compiled_output, pass_stats, precision)


class CompilerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Model Compiler | Workbench")
        self.root.geometry("1040x700")
        self.root.minsize(850, 580)
        self.result: CompilationResult | None = None
        self.model_var = tk.StringVar(value="MLP demo")
        self.precision_var = tk.StringVar(value="FP32")
        self.status_var = tk.StringVar(value="Ready to compile a workload")
        self.pass_vars = {
            name: tk.BooleanVar(value=True)
            for name in (
                "Constant folding",
                "Dead-node elimination",
                "Redundant operation elimination",
                "Conv-BN-ReLU fusion",
            )
        }
        self._build_layout()

    def _build_layout(self) -> None:
        self.root.configure(bg="#09141b")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Workbench.TCombobox", fieldbackground="#1a3039", background="#1a3039", foreground="#e6f2ed", bordercolor="#2c4b54", arrowcolor="#e6a26b")
        style.configure("Workbench.TCheckbutton", background="#12242d", foreground="#c6d8d3", font=("Segoe UI", 9))
        style.map("Workbench.TCheckbutton", background=[("active", "#19333c")], foreground=[("active", "#ffffff")])
        style.configure("Vertical.TScrollbar", troughcolor="#10242b", background="#31525a", bordercolor="#10242b", arrowcolor="#d8a16f")

        header = tk.Frame(self.root, bg="#102932", height=78)
        header.pack(fill="x")
        header.pack_propagate(False)
        brand = tk.Frame(header, bg="#102932")
        brand.pack(side="left", padx=26, pady=14)
        tk.Label(brand, text="AI MODEL COMPILER", bg="#102932", fg="#f4f7f3", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(brand, text="GRAPH OPTIMIZATION WORKBENCH  /  CPU REFERENCE RUNTIME", bg="#102932", fg="#8eb5ad", font=("Consolas", 8)).pack(anchor="w", pady=(3, 0))
        tk.Label(header, text="●  LOCAL SESSION", bg="#102932", fg="#83d6ae", font=("Consolas", 9, "bold")).pack(side="right", padx=26)

        body = tk.Frame(self.root, bg="#09141b")
        body.pack(fill="both", expand=True, padx=18, pady=18)
        controls = tk.Frame(body, bg="#12242d", width=270, highlightbackground="#1e3a43", highlightthickness=1)
        controls.pack(side="left", fill="y", padx=(0, 18))
        controls.pack_propagate(False)
        tk.Label(controls, text="COMPILATION CONTROL", bg="#12242d", fg="#83d6ae", font=("Consolas", 9, "bold")).pack(anchor="w", padx=18, pady=(20, 16))
        tk.Label(controls, text="WORKLOAD", bg="#12242d", fg="#72949a", font=("Consolas", 8, "bold")).pack(anchor="w", padx=18)
        ttk.Combobox(controls, textvariable=self.model_var, values=("MLP demo", "CNN demo"), state="readonly", style="Workbench.TCombobox").pack(fill="x", padx=18, pady=(5, 14))
        tk.Label(controls, text="PRECISION MODE", bg="#12242d", fg="#72949a", font=("Consolas", 8, "bold")).pack(anchor="w", padx=18)
        ttk.Combobox(controls, textvariable=self.precision_var, values=("FP32", "FP16", "INT8"), state="readonly", style="Workbench.TCombobox").pack(fill="x", padx=18, pady=(5, 20))
        tk.Label(controls, text="OPTIMIZATION PASSES", bg="#12242d", fg="#72949a", font=("Consolas", 8, "bold")).pack(anchor="w", padx=18, pady=(0, 6))
        passes = tk.Frame(controls, bg="#12242d")
        passes.pack(fill="x", padx=14)
        for name, variable in self.pass_vars.items():
            ttk.Checkbutton(passes, text=name, variable=variable, style="Workbench.TCheckbutton").pack(anchor="w", pady=4)

        tk.Button(controls, text="▶  COMPILE WORKLOAD", command=self.compile, bg="#d36f48", fg="#ffffff", activebackground="#e5895e", activeforeground="#ffffff", relief="flat", borderwidth=0, font=("Segoe UI", 10, "bold"), cursor="hand2", pady=10).pack(fill="x", padx=18, pady=(24, 8))
        tk.Button(controls, text="◷  Benchmark graph", command=self.run_benchmark, bg="#1b3942", fg="#d9ebe5", activebackground="#28515a", activeforeground="#ffffff", relief="flat", borderwidth=0, font=("Segoe UI", 9), cursor="hand2", pady=8).pack(fill="x", padx=18, pady=3)
        tk.Button(controls, text="⇩  Export IR JSON", command=self.export_ir, bg="#1b3942", fg="#d9ebe5", activebackground="#28515a", activeforeground="#ffffff", relief="flat", borderwidth=0, font=("Segoe UI", 9), cursor="hand2", pady=8).pack(fill="x", padx=18, pady=3)
        tk.Button(controls, text="Clear console", command=self.clear_output, bg="#12242d", fg="#77999d", activebackground="#19333c", activeforeground="#ffffff", relief="flat", borderwidth=0, font=("Segoe UI", 9), cursor="hand2", pady=7).pack(fill="x", padx=18, pady=3)
        self.status_label = tk.Label(controls, textvariable=self.status_var, bg="#12242d", fg="#83d6ae", anchor="w", wraplength=220, font=("Segoe UI", 9))
        self.status_label.pack(anchor="w", padx=18, pady=(18, 0))

        main = tk.Frame(body, bg="#09141b")
        main.pack(side="left", fill="both", expand=True)
        heading = tk.Frame(main, bg="#09141b")
        heading.pack(fill="x", pady=(0, 13))
        tk.Label(heading, text="Compilation overview", bg="#09141b", fg="#f4f7f3", font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(heading, text="CUSTOM IR  /  LIVE", bg="#09141b", fg="#83d6ae", font=("Consolas", 9, "bold")).pack(side="right", pady=6)

        metrics = tk.Frame(main, bg="#09141b")
        metrics.pack(fill="x", pady=(0, 14))
        self.metric_vars = {}
        for title, key, accent in (("IR NODES", "nodes", "#d36f48"), ("TENSORS", "tensors", "#75b9ae"), ("MAX ERROR", "error", "#e5b567"), ("LATENCY", "latency", "#8a9ee8")):
            card = tk.Frame(metrics, bg="#12242d", highlightbackground="#1e3a43", highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=(0, 8))
            tk.Frame(card, bg=accent, height=3).pack(fill="x")
            tk.Label(card, text=title, bg="#12242d", fg="#72949a", font=("Consolas", 8, "bold")).pack(anchor="w", padx=12, pady=(10, 1))
            value = tk.StringVar(value="—")
            self.metric_vars[key] = value
            tk.Label(card, textvariable=value, bg="#12242d", fg="#f4f7f3", font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=12, pady=(0, 10))

        content = tk.Frame(main, bg="#09141b")
        content.pack(fill="both", expand=True)
        result_panel = tk.Frame(content, bg="#0d2027", highlightbackground="#1e3a43", highlightthickness=1)
        result_panel.pack(side="left", fill="both", expand=True)
        tk.Label(result_panel, text="  COMPILER CONSOLE", bg="#0d2027", fg="#83d6ae", anchor="w", font=("Consolas", 9, "bold")).pack(fill="x", pady=(10, 3))
        self.output = tk.Text(result_panel, wrap="word", background="#0d2027", foreground="#d9f0e5", insertbackground="#ffffff", font=("Cascadia Mono", 10), padx=14, pady=10, relief="flat")
        self.output.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(result_panel, orient="vertical", command=self.output.yview, style="Vertical.TScrollbar")
        scrollbar.pack(side="right", fill="y", padx=(0, 4), pady=(0, 8))
        self.output.configure(yscrollcommand=scrollbar.set)

    def _write(self, text: str) -> None:
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def compile(self) -> None:
        self.status_var.set("Compiling...")
        self.output.delete("1.0", "end")
        try:
            enabled = tuple(name for name, variable in self.pass_vars.items() if variable.get())
            self.result = compile_workload(self.model_var.get(), self.precision_var.get(), enabled)
            result = self.result
            max_error = (result.compiled_output.float() - result.reference_output.float()).abs().max().item()
            self.metric_vars["nodes"].set(str(len(result.graph.nodes)))
            self.metric_vars["tensors"].set(str(len(result.graph.tensors)))
            self.metric_vars["error"].set(f"{max_error:.2e}")
            self.metric_vars["latency"].set("—")
            self._write(f"WORKLOAD  {self.model_var.get()}")
            self._write(f"PRECISION {result.precision}")
            self._write(f"IR       {len(result.graph.nodes)} nodes, {len(result.graph.tensors)} tensors")
            self._write(f"INPUT    {result.graph.inputs} -> OUTPUT {result.graph.outputs}")
            self._write("\nPASS STATISTICS")
            self._write(json.dumps(result.pass_stats, indent=2))
            self._write(f"\nMAX OUTPUT ERROR  {max_error:.8f}")
            self._write("\nGraph is valid and ready for benchmarking.")
            self.status_var.set("Compilation complete")
        except Exception as error:
            self.status_var.set("Compilation failed")
            messagebox.showerror("Compilation error", str(error))

    def run_benchmark(self) -> None:
        if self.result is None:
            self.compile()
        if self.result is None:
            return
        measured = benchmark(self.result.graph, {self.result.graph.inputs[0]: self.result.example_input}, warmup=5, runs=30)
        self.metric_vars["latency"].set(f"{measured.mean_ms:.3f} ms")
        self._write("\nBENCHMARK")
        self._write(json.dumps(measured.to_dict(), indent=2))
        self.status_var.set("Benchmark complete")

    def export_ir(self) -> None:
        if self.result is None:
            self.compile()
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export compiler IR", defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            initialfile=f"{self.model_var.get().lower().replace(' ', '_')}_ir.json",
        )
        if path:
            write_graph_json(self.result.graph, path)
            self._write(f"\nEXPORTED IR\n{path}")
            self.status_var.set("IR exported")

    def clear_output(self) -> None:
        self.output.delete("1.0", "end")
        self.status_var.set("Ready to compile a workload")


def launch() -> None:
    root = tk.Tk()
    CompilerApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()