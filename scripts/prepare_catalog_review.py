"""Prepare a read-only catalog audit and a standalone offline gallery.

python -m scripts.prepare_catalog_review --bundle build/real-catalog/projected-v2 \
    --output-dir build/review/catalog
"""

import argparse
from pathlib import Path

from app.catalog_preparation.catalog_review import prepare_catalog_review
from app.catalog_preparation.workflow import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "outfitseed.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_catalog_review(args.bundle, args.csv, args.output_dir)
    print(
        f"Reviewed {report['row_count']} rows; issues={report['severity_counts']}. "
        f"Gallery: {args.output_dir / 'gallery.html'}"
    )


if __name__ == "__main__":
    main()
