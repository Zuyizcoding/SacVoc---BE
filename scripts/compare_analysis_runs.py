"""Compare complete F1/F2 runs with the same dataset, inputs, and selected split."""

import argparse
from pathlib import Path

from app.evaluation.comparison import compare_analysis_runs
from app.evaluation.comparison_report import write_comparison_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compare_analysis_runs(args.baseline, args.candidate)
    write_comparison_report(args.output_dir, report)
    print(report["interpretation"])
    print(f"Report: {args.output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
