"""Inspect an analysis manifest or retrieval snapshot without AI or DB writes."""

import argparse
from pathlib import Path

from app.evaluation.preflight import (
    preflight_analysis,
    preflight_retrieval,
    write_preflight_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["analysis", "retrieval"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inspect = preflight_analysis if args.kind == "analysis" else preflight_retrieval
    report = inspect(args.dataset)
    write_preflight_report(args.output_dir, report)
    print(f"Ready: {report['ready']}; report: {args.output_dir / 'report.json'}")
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
