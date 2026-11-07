"""Prepare a fresh real catalog, blocking encoding on incompatible queries.

Run from the backend root in the optional catalog environment:
python -m scripts.prepare_real_catalog --output build/real-catalog/run-01 \
    --model-directory build/catalog-model

Downloads only public catalog images. The model must already be downloaded at
the revision documented in app.catalog_preparation.encoder. Existing output
directories are refused. Exit 2 means incompatible encoder; 3 means no images.
A blocked run still preserves the seeded, unencoded DB and diagnostic reports.
"""

import argparse
from pathlib import Path

from app.catalog_preparation.encoder import PinnedFashionClip
from app.catalog_preparation.workflow import PROJECT_ROOT, prepare_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "outfitseed.csv")
    parser.add_argument(
        "--query-vectors", type=Path, default=PROJECT_ROOT / "data/query_vectors.json"
    )
    parser.add_argument(
        "--query-metadata",
        type=Path,
        default=PROJECT_ROOT / "data/query_vectors.meta.json",
    )
    args = parser.parse_args()
    summary = prepare_catalog(
        args.csv,
        args.query_vectors,
        args.query_metadata,
        args.output,
        lambda: PinnedFashionClip(args.model_directory),
    )
    print(
        f"Status: {summary['status']}; encoded {summary['encoded_rows']}/"
        f"{summary['csv_rows']} rows. Reports: {args.output.resolve()}"
    )
    if summary["status"] == "blocked_incompatible_encoder":
        raise SystemExit(2)
    if summary["status"] == "blocked_no_available_images":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
