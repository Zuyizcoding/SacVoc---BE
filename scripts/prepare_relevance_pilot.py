"""Prepare a blinded, offline 20-scenario F3 pilot; all human grades stay blank.

python -m scripts.prepare_relevance_pilot --bundle build/real-catalog/projected-v2 \
    --catalog-audit build/review/catalog/report.json --output-dir build/review/pilot
"""

import argparse
from pathlib import Path

from app.catalog_preparation.pilot import prepare_relevance_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--catalog-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_relevance_pilot(args.bundle, args.catalog_audit, args.output_dir)
    print(
        f"Prepared {report['scenario_count']} dev scenarios, "
        f"{report['tasks_per_reviewer']} blank tasks per reviewer."
    )


if __name__ == "__main__":
    main()
