"""Verify an existing corrected bundle without models, Gemini or DB writes.

python -m scripts.verify_embedding_bundle --bundle build/real-catalog/projected-v2 \
    --output-dir build/real-catalog/projected-v2-verification

Checks manifest hashes and preserved baseline inputs, then tests runtime/export
filter parity and deterministic ranking for all 240 profiles. Reports are saved
in a separate new directory. This does not measure recommendation accuracy.
"""

import argparse
from pathlib import Path

from app.catalog_preparation.encoder import file_sha256
from app.catalog_preparation.integrity import verify_manifest
from app.catalog_preparation.validation import verify_serving_paths
from app.catalog_preparation.workflow import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    integrity = verify_manifest(args.bundle)
    before = file_sha256(args.bundle / "catalog.db")
    report = verify_serving_paths(args.bundle)
    if file_sha256(args.bundle / "catalog.db") != before:
        raise ValueError("Read-only serving verification changed the DB")
    report["integrity"] = integrity
    report["database_unchanged"] = True
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "report.json", report)
    print(
        f"Verified {report['profiles_checked']} profiles; "
        f"DB and baseline unchanged. Report: {args.output_dir / 'report.json'}"
    )


if __name__ == "__main__":
    main()
