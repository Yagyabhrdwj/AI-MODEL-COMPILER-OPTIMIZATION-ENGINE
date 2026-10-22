from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "AI_MODEL_COMPILER_STATUS.pdf"


def bullet_items(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BodyText"])) for item in items],
        bulletType="bullet",
        leftIndent=18,
    )


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, leading=27, spaceAfter=8))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=20))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#0f4c5c"), spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Code"], fontSize=8.5, leading=11, backColor=colors.HexColor("#f1f5f9"), borderPadding=7, spaceBefore=5, spaceAfter=8))


def section(title: str, body: str):
    return [Paragraph(title, styles["Section"]), Paragraph(body, styles["BodyText"]), Spacer(1, 4)]


def build_pdf() -> None:
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="AI Model Compiler and Optimization Engine - Project Status",
        author="AI Model Compiler project",
    )
    story = [
        Paragraph("AI Model Compiler & Optimization Engine", styles["ReportTitle"]),
        Paragraph(f"Implementation status and code update report | {date.today().isoformat()}", styles["Subtitle"]),
        Paragraph("Executive status", styles["Section"]),
        Paragraph(
            "The repository now contains a tested compiler core for importing selected PyTorch models into a custom intermediate representation, validating the graph, applying graph optimizations, executing the result through a reference backend, and measuring latency. The work is a functional prototype; the full research program is not yet complete.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]

    status_table = Table([
        ["Area", "Current status"],
        ["Custom IR", "Implemented and validated"],
        ["PyTorch importer", "MLP and controlled CNN subset implemented"],
        ["Optimization passes", "Constant folding, dead-node, redundant-op, and Conv-BN-ReLU fusion"],
        ["Reference backend", "Implemented with PyTorch tensor kernels"],
        ["Precision", "FP16 and symmetric weight-only INT8 implemented"],
        ["Visualization", "DOT output implemented"],
        ["Benchmarking", "Repeated latency and throughput runner implemented"],
        ["GUI", "Tkinter workbench for compile, benchmark, and IR export"],
        ["Automated tests", "9 tests passing at report generation"],
    ], colWidths=[48 * mm, 125 * mm])
    status_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [status_table, PageBreak()]

    story += section("1. Architecture implemented", "The current pipeline is:")
    story.append(Paragraph("PyTorch model -> torch.fx extraction -> custom Graph/Node/Tensor IR -> validation -> optimization pass -> reference backend -> numerical comparison and benchmark", styles["CodeBlock"]))
    story += section("2. Code updates made", "The following modules were added or expanded during implementation:")
    story.append(bullet_items([
        "compiler/ir: graph ownership, tensor metadata, producer/consumer links, boundary checks, topological ordering, and cycle detection.",
        "compiler/frontend/pytorch_importer.py: FX import for Linear, ReLU, Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, Flatten, Add, and Mul.",
        "compiler/optimizer: pass manager, constant folding, dead-node elimination, scalar identity elimination, and inference Conv-BatchNorm-ReLU fusion.",
        "compiler/backend/reference.py: execution of the imported and fused IR using PyTorch reference kernels.",
        "compiler/quantization/precision.py: FP16 conversion plus symmetric weight-only INT8 quantization with retained scales.",
        "visualization/dot.py: Graphviz DOT generation without requiring Graphviz installation.",
        "benchmarks/runner.py: warmups, repeated timings, mean/median/p95 latency, throughput, and runtime metadata.",
        "tests/: import, execution, malformed operator diagnostics, optimization, fusion, precision, visualization, and benchmark regression tests.",
        "compiler/gui.py: Tkinter workbench for selecting workloads, precision, passes, benchmarking, and exporting IR JSON.",
    ]))
    story += section("3. Correctness evidence", "The most important checks compare reference PyTorch execution against execution of the custom IR. The MLP and CNN checks pass with zero maximum error in the tested FP32 cases. Conv-BatchNorm-ReLU fusion is checked with dtype-appropriate tolerance, and the FP16 path verifies reduced-precision output and constant metadata.")
    story.append(Paragraph("Latest validation result: 12 tests passed; static diagnostics reported no errors; Python bytecode compilation completed successfully.", styles["CodeBlock"]))
    story += section("4. Benchmarking currently available", "The benchmark runner is a measurement utility. It records warmup count, run count, batch size, mean latency, median latency, p95 latency, throughput, Python version, PyTorch version, and device. It does not claim an optimization speedup until controlled baseline and optimized experiments are run.")
    story.append(Paragraph("Run: python -m scripts.benchmark_demo", styles["CodeBlock"]))
    story.append(PageBreak())
    story += section("5. Important design decisions", "The project owns the IR and optimization logic. PyTorch FX is used for extraction and PyTorch kernels are used only as a transparent reference backend. This preserves the compiler research focus while avoiding premature hand-written numerical kernels. Unsupported operators raise explicit diagnostics instead of silently producing an incomplete graph.")
    story += section("6. Current limitations", "The current repository is not yet the complete final research system. It does not yet include INT8 activation calibration, ONNX export or runtime comparison, peak-memory measurement, serialized model-size measurement, accuracy evaluation on CIFAR-10, ResNet-18 or MobileNet experiments, automated ablation reports, benchmark plots, or the final academic report.")
    story += section("7. Recommended next updates", "The next implementation sequence should be:")
    story.append(bullet_items([
        "Add graph serialization and stable experiment configuration files.",
        "Add an INT8 investigation with calibration, numerical-error reporting, and explicit accuracy trade-offs.",
        "Add benchmark result persistence to CSV/JSON and generate plots for baseline versus each pass.",
        "Add ResNet-18 and MobileNet controlled workload import tests, then run ablation experiments.",
        "Add optional ONNX Runtime comparison while keeping custom IR transformations independent.",
        "Write the research report from measured results, including neutral or failed optimization cases.",
    ]))
    story += section("8. Reproduction commands", "From the repository root:")
    story.append(Paragraph("python -m pytest -q\npython -m compiler.demo\npython -m scripts.benchmark_demo\npython -m compileall -q compiler benchmarks visualization scripts tests", styles["CodeBlock"]))
    story += section("Source references", "The living technical plan is in docs/PROJECT_BLUEPRINT.md. The implementation overview and setup commands are in README.md. This PDF is generated by docs/generate_status_pdf.py so it can be refreshed as the code changes.")

    document.build(story)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)