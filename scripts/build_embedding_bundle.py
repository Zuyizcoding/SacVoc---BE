"""Build a corrected, separate F3 bundle; never switch application settings.

Example:
python -m scripts.build_embedding_bundle --output build/real-catalog/projected-v2 \
    --model-directory build/catalog-model

Uses the optional requirements-catalog environment and pinned local weights.
Public image downloads use normal TLS. A new output directory is mandatory.
The baseline's vectors, CSV and manifests are hashed and left unchanged.
"""

import argparse
from pathlib import Path

from app.catalog_preparation.bundle import build_corrected_bundle
from app.catalog_preparation.encoder import PinnedFashionClip
from app.catalog_preparation.workflow import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    args = parser.parse_args()
    summary = build_corrected_bundle(
        PROJECT_ROOT / "outfitseed.csv",
        PROJECT_ROOT / "data/query_vectors.json",
        PROJECT_ROOT / "data/query_vectors.meta.json",
        args.output,
        lambda: PinnedFashionClip(args.model_directory),
        preserved_files=(
            PROJECT_ROOT / "data/outfit_vectors.meta.json",
            PROJECT_ROOT / "app/services/recommendation/query_builder.py",
        ),
    )
    print(
        f"{summary['status']}: {summary['encoded_rows']}/{summary['csv_rows']} rows; "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
