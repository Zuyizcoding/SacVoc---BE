"""CLI for isolated F1/F2 evaluation; see docs/evaluation.md for the contracts."""

import argparse
import asyncio
from pathlib import Path

from app.evaluation.runner import run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; must not exist"
    )
    parser.add_argument("--provider", choices=["synthetic", "gemini"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    args = parser.parse_args()
    report = asyncio.run(
        run_dataset(
            args.dataset, args.output_dir, provider=args.provider, split=args.split
        )
    )
    print(report["interpretation"])
    print(f"Report: {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
