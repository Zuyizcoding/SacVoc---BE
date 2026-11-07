"""Check the actual F3 FastAPI routes against an isolated copy of a v2 bundle.

python -m scripts.verify_bundle_api --bundle build/real-catalog/projected-v2 \
    --output-dir build/review/api
"""

import argparse
from pathlib import Path

from app.catalog_preparation.api_verification import run_isolated_api_verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_isolated_api_verification(args.bundle, args.output_dir)
    print(
        f"Verified {report['profiles_checked']} API profiles and "
        f"{report['detail_rows_checked']} outfit details; source DB unchanged."
    )


if __name__ == "__main__":
    main()
