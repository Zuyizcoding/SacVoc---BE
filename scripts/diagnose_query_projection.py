"""Diagnose a possible historical double text projection; never encode images.

Intuition: CLIP's projected text features and hidden text features may both be
512-dimensional. A compatibility adapter cannot use dimension alone to decide
whether to apply text_projection. Applying it again changes the shared geometry.

This diagnostic compares that extra-projection hypothesis to all saved queries.
Even a perfect match is NOT permission to encode images or declare the normal
encoder compatible. It identifies a likely historical generation path only.
"""

import argparse
import json
from pathlib import Path

from app.catalog_preparation.compatibility import (
    read_reference_queries,
    verify_compatibility,
)
from app.catalog_preparation.encoder import PinnedFashionClip
from app.catalog_preparation.workflow import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoder = PinnedFashionClip(args.model_directory)

    class ExtraTextProjection:
        def encode_text(self, texts):
            import torch

            raw = encoder.encode_text(texts)
            with torch.inference_mode():
                return encoder.model.text_projection(torch.from_numpy(raw)).numpy()

    reference = read_reference_queries(
        PROJECT_ROOT / "data/query_vectors.json",
        PROJECT_ROOT / "data/query_vectors.meta.json",
    )
    result = verify_compatibility(ExtraTextProjection(), reference)
    result["diagnostic_only"] = True
    result["hypothesis"] = "text features were projected a second time"
    # Refuse overwrite so each diagnostic remains reviewable.
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2)
    print({key: value for key, value in result.items() if key != "queries"})


if __name__ == "__main__":
    main()
