"""Offline outfit image embedding for F3.

Usage:
    cd CS4153.Q24.CTTT---BE
    python -m scripts.encode_outfits

Optional:
    python -m scripts.encode_outfits --force
    python -m scripts.encode_outfits --fail-on-mismatch
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import SessionLocal, init_db
from app.db.models import Outfit
from app.services.recommendation.types import PROJECTED_EMBEDDING_VERSION
from app.utils.fashion_clip_compat import clip_features_to_numpy, load_fashion_clip


def _safe_console_text(value: object) -> str:
    text = str(value)
    return text.encode("ascii", errors="replace").decode("ascii")


def _default_csv_path() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "outfitseed.csv")
    )


def _read_csv_keys(csv_path: str) -> tuple[set[str], set[str], int]:
    urls: set[str] = set()
    names: set[str] = set()
    row_count = 0
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            name = (row.get("name") or "").strip()
            url = (row.get("shopee_url") or "").strip()
            if name:
                names.add(name)
            if url:
                urls.add(url)
    return urls, names, row_count


def _in_csv(*, name: str, url: str, csv_urls: set[str], csv_names: set[str]) -> bool:
    name = (name or "").strip()
    url = (url or "").strip()
    return (bool(url) and url in csv_urls) or (bool(name) and name in csv_names)


def download_image(url: str) -> Image.Image | None:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        print(f"  WARN download failed {url}: {exc}")
        return None


def _build_image_encoder():
    """Create real image encoder: prefer FashionCLIP, fallback to HF CLIP."""
    try:
        print("Loading FashionCLIP model...")
        fclip = load_fashion_clip("fashion-clip")
        print("Model loaded (backend=fashion_clip)")

        def _encode(image: Image.Image) -> np.ndarray:
            return fclip.encode_images([image], batch_size=1)[0]

        return _encode, {
            "backend": "fashion_clip",
            "model": "fashion-clip",
            "version": PROJECTED_EMBEDDING_VERSION,
        }
    except Exception as exc:
        print(f"FashionCLIP unavailable, fallback to transformers CLIP: {exc}")

    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        print(
            "ERROR: missing image-encoder dependencies. "
            "Install fashion-clip or torch + transformers. "
            f"Detail: {exc}"
        )
        return None, None

    print("Loading transformers CLIP (openai/clip-vit-base-patch32)...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("Model loaded (backend=hf_clip)")

    def _encode(image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            inputs = processor(images=[image], return_tensors="pt")
            raw = model.get_image_features(**inputs)
            features = clip_features_to_numpy(raw)
        return features[0]

    return _encode, {
        "backend": "hf_clip",
        "model": "openai/clip-vit-base-patch32",
        "version": PROJECTED_EMBEDDING_VERSION,
    }


def _write_manifest(meta: dict[str, str], success_count: int, total_count: int) -> None:
    manifest = {
        "backend": meta["backend"],
        "model": meta["model"],
        "version": meta["version"],
        "encoded_outfits": success_count,
        "requested_outfits": total_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "outfit_vectors.meta.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote manifest -> {output_path}")


def encode_all() -> None:
    parser = argparse.ArgumentParser(description="Encode outfit vectors for F3")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode all outfits and overwrite existing vector_blob",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=_default_csv_path(),
        help="CSV path to cross-check DB outfits",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit if DB outfits differ from CSV (extra rows present in DB)",
    )
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    try:
        image_encoder, meta = _build_image_encoder()
        if image_encoder is None:
            return

        csv_path = os.path.abspath(args.csv_path)
        if os.path.exists(csv_path):
            try:
                csv_urls, csv_names, csv_rows = _read_csv_keys(csv_path)
                db_all = db.query(Outfit.id, Outfit.name, Outfit.shopee_url).all()
                extras = [
                    (oid, name, url)
                    for (oid, name, url) in db_all
                    if not _in_csv(
                        name=name or "",
                        url=url or "",
                        csv_urls=csv_urls,
                        csv_names=csv_names,
                    )
                ]
                if extras:
                    print(
                        f"WARNING: DB has {len(db_all)} outfits but CSV has {csv_rows}; "
                        f"{len(extras)} extra outfit(s) in DB"
                    )
                    for item in extras[:10]:
                        print(_safe_console_text(f"  EXTRA {item}"))
                    if args.fail_on_mismatch:
                        print("ERROR: mismatch detected (--fail-on-mismatch). Abort.")
                        return
            except Exception as exc:
                print(f"WARNING: CSV cross-check failed: {exc}")

        query = db.query(Outfit)
        if not args.force:
            query = query.filter(Outfit.vector_blob.is_(None))
        outfits = query.all()

        if not outfits:
            print("INFO: no outfit needs encoding.")
            return

        print(f"Encoding {len(outfits)} outfits (backend={meta['backend']})...")

        success_count = 0
        for i, outfit in enumerate(outfits, 1):
            safe_name = _safe_console_text(outfit.name)
            print(f"  [{i}/{len(outfits)}] {safe_name}...", end=" ")

            image = download_image(outfit.image_url)
            if image is None:
                print("SKIP")
                continue

            try:
                encoded = image_encoder(image)
            except Exception as exc:
                print(f"SKIP (encode error: {exc})")
                continue

            vector = encoded.astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm == 0:
                print("SKIP (zero-norm vector)")
                continue

            vector = vector / norm
            outfit.vector_blob = vector.tobytes()
            outfit.embedding_backend = meta["backend"]
            outfit.embedding_model = meta["model"]
            outfit.embedding_version = meta["version"]
            outfit.embedding_generated_at = datetime.now(timezone.utc)
            success_count += 1
            print("OK")

        db.commit()
        print(f"OK encode done: {success_count}/{len(outfits)} outfits.")
        _write_manifest(meta, success_count, len(outfits))

    finally:
        db.close()


if __name__ == "__main__":
    encode_all()
