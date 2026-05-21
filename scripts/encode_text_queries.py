"""Pre-encode F3 text query vectors."""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.services.outfit_service import (
    BODY_KEYWORDS,
    OCCASION_KEYWORDS,
    SEASON_KEYWORDS,
    build_outfit_query,
)


EMBEDDING_SCHEMA_VERSION = "2026-05-mens-v1"


def _all_query_templates() -> list[str]:
    queries: list[str] = []
    for season in SEASON_KEYWORDS:
        for body_shape in BODY_KEYWORDS:
            for occasion in OCCASION_KEYWORDS:
                queries.append(build_outfit_query(season, body_shape, occasion))
    return queries


def _query_signature(queries: list[str]) -> str:
    payload = "\n".join(sorted(queries)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_numpy_features(raw, projection_layer=None) -> np.ndarray:
    """Handle both tensor output and model-output object (transformers v5+)."""
    if hasattr(raw, "cpu"):
        return raw.cpu().numpy()

    pooled = getattr(raw, "pooler_output", None)
    if pooled is None:
        raise TypeError(f"Unsupported feature output type: {type(raw)}")

    if projection_layer is not None:
        in_features = getattr(projection_layer, "in_features", None)
        if in_features is None or pooled.shape[-1] == in_features:
            pooled = projection_layer(pooled)

    return pooled.cpu().numpy()


def encode_queries() -> None:
    queries = _all_query_templates()
    text_embeddings = None

    try:
        from fashion_clip.fashion_clip import FashionCLIP

        print("Loading FashionCLIP...")
        fclip = FashionCLIP("fashion-clip")
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
            print("ERROR: Missing encoder deps. Install torch + transformers or fashion-clip.")
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
            text_embeddings = _to_numpy_features(raw, projection_layer=getattr(model, "text_projection", None))
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

    with open(os.path.join(output_dir, "query_vectors.meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **(meta or {}),
                "query_count": len(queries),
                "query_signature": _query_signature(queries),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(output)} vectors -> {output_path}")


if __name__ == "__main__":
    encode_queries()
