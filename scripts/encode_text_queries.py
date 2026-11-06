"""Pre-encode F3 text query vectors."""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.services.recommendation.query_builder import (
    _all_query_templates,
    _current_query_signature,
)
from app.services.recommendation.types import EMBEDDING_SCHEMA_VERSION
from app.utils.fashion_clip_compat import clip_features_to_numpy, load_fashion_clip


def encode_queries() -> None:
    queries = _all_query_templates()
    text_embeddings = None

    try:
        print("Loading FashionCLIP...")
        fclip = load_fashion_clip("fashion-clip")
        print(f"Encoding {len(queries)} queries with FashionCLIP...")
        text_embeddings = fclip.encode_text(queries, batch_size=32)
        meta = {
            "backend": "fashion_clip",
            "model": "fashion-clip",
            "version": EMBEDDING_SCHEMA_VERSION,
        }
    except Exception as exc:
        print(f"FashionCLIP unavailable, fallback to transformers CLIP: {exc}")
        meta = None

    if text_embeddings is None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception:
            print(
                "ERROR: Missing encoder deps. Install torch + transformers or fashion-clip."
            )
            return

        print("Loading transformers CLIP (openai/clip-vit-base-patch32)...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        print(f"Encoding {len(queries)} queries with transformers CLIP...")
        with torch.no_grad():
            inputs = processor(
                text=queries,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            raw = model.get_text_features(**inputs)
            text_embeddings = clip_features_to_numpy(
                raw, projection_layer=getattr(model, "text_projection", None)
            )
        meta = {
            "backend": "hf_clip",
            "model": "openai/clip-vit-base-patch32",
            "version": EMBEDDING_SCHEMA_VERSION,
        }

    norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / norms

    output = {}
    for query, vector in zip(queries, text_embeddings):
        output[query] = vector.astype(np.float32).tolist()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "query_vectors.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    with open(
        os.path.join(output_dir, "query_vectors.meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                **(meta or {}),
                "query_count": len(queries),
                "query_signature": _current_query_signature(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(output)} vectors -> {output_path}")


if __name__ == "__main__":
    encode_queries()
