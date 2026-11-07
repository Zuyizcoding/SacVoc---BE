"""Compare four F3 ranking variants on a frozen, independently judged snapshot."""

import argparse
from pathlib import Path

from app.evaluation.retrieval import evaluate_snapshot, write_ablation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_snapshot(
        args.snapshot, split=args.split, seed=args.seed, k=args.top_k
    )
    write_ablation_report(args.output_dir, report)
    print(report["interpretation"])
    print(f"Report: {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
