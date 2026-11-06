"""Offline dummy encoder for Outfit vectors (no FashionCLIP required).

Purpose:
- Unblock F3 recommendation which filters on `Outfit.vector_blob IS NOT NULL`.
- Generate deterministic 512-d float32 vectors (normalized) using only numpy.

Run:
    cd src/backend
    python -m scripts.encode_outfits_dummy

Notes:
- This does NOT provide true semantic similarity; it only ensures the API returns outfits.
- If you want real FashionCLIP vectors, use `scripts/encode_outfits.py` with compatible deps.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.db.database import SessionLocal, init_db
from app.db.models import Outfit

EMBEDDING_SCHEMA_VERSION = "2026-05-dummy"


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _make_vector(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


def encode_all() -> None:
    init_db()
    db = SessionLocal()

    try:
        outfits = db.query(Outfit).filter(Outfit.vector_blob.is_(None)).all()
        if not outfits:
            print("INFO: All outfits already have vector_blob; nothing to do.")
            return

        updated = 0
        for outfit in outfits:
            key = f"{outfit.shopee_url}|{outfit.image_url}|{outfit.name}".strip("|")
            seed = _seed_from_text(key)
            outfit.vector_blob = _make_vector(seed)
            outfit.embedding_backend = "dummy"
            outfit.embedding_model = "sha256-seeded-rng"
            outfit.embedding_version = EMBEDDING_SCHEMA_VERSION
            outfit.embedding_generated_at = datetime.now(timezone.utc)
            updated += 1

        db.commit()
        print(f"OK Dummy encode done: updated={updated}")
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "outfit_vectors.meta.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "backend": "dummy",
                    "model": "sha256-seeded-rng",
                    "version": EMBEDDING_SCHEMA_VERSION,
                    "encoded_outfits": updated,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    finally:
        db.close()


if __name__ == "__main__":
    encode_all()
