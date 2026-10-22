from __future__ import annotations

from pathlib import Path

from visualization.benchmarks import plot_benchmark_csv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    input_path = ROOT / "results" / "ablation.csv"
    output_path = ROOT / "results" / "ablation.png"
    try:
        plot_benchmark_csv(input_path, output_path)
    except RuntimeError as error:
        print(error)
        return
    print(output_path)


if __name__ == "__main__":
    main()