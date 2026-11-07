"""Validate two completed reviewer files and report/resolve relevance disagreements."""

import argparse
from pathlib import Path

from app.evaluation.annotation_review import review_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--resolutions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = review_annotations(
            args.packet,
            args.first,
            args.second,
            args.output_dir,
            resolution_path=args.resolutions,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Ready: {report['ready']}; disagreements: {report['disagreement_count']}")
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
