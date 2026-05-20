"""FashionCLIP Encoding — Chạy offline trên máy dev.

Tính năng #29:
  1. Load FashionCLIP model (fashion-clip)
  2. Đọc 80 ảnh outfit từ URLs trong DB
  3. Encode mỗi ảnh → vector 512 dims (float32)
  4. Lưu vector dạng BLOB vào cột `vector_blob` trong bảng Outfit

Cách chạy:
    pip install fashion-clip Pillow requests
    cd src/backend
    python -m scripts.encode_outfits

Re-encode (ghi đè vector hiện có):
    python -m scripts.encode_outfits --force

SAU KHI CHẠY:
  - Mỗi outfit row sẽ có vector_blob (2048 bytes)
  - Server chỉ cần numpy để tính cosine similarity
  - Server KHÔNG cần install fashion-clip (model nặng ~400MB)

LƯU Ý QUAN TRỌNG:
  - Script này KHÔNG chạy trên server Railway (không đủ RAM)
  - Chỉ chạy trên máy dev có ít nhất 4GB RAM
  - Nếu ảnh URL lỗi → bỏ qua, in warning
"""
import sys
import os

import argparse
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import requests
from io import BytesIO
from PIL import Image

from app.db.database import SessionLocal, init_db
from app.db.models import Outfit


def _default_csv_path() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outfitseed.csv"))


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
    """Tải ảnh từ URL — trả None nếu lỗi."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"  ⚠️  Lỗi tải ảnh {url}: {e}")
        return None


def encode_all() -> None:
    """Encode tất cả outfit chưa có vector → lưu BLOB vào DB."""
    # Import FashionCLIP (chỉ cần trên máy dev)
    try:
        from fashion_clip.fashion_clip import FashionCLIP
    except ImportError:
        print("ERROR: fashion-clip is not installed. Run: pip install fashion-clip")
        return

    parser = argparse.ArgumentParser(description="Encode outfit vectors with FashionCLIP")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode all outfits and overwrite existing vector_blob",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=_default_csv_path(),
        help="CSV path to cross-check DB outfits (default: src/backend/outfitseed.csv)",
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
        # Khởi tạo model (lần đầu sẽ download ~400MB)
        print("Loading FashionCLIP model...")
        try:
            fclip = FashionCLIP("fashion-clip")
        except TypeError as e:
            # Known issue: fashion-clip==0.3.1 is incompatible with transformers>=5
            msg = str(e)
            if "use_auth_token" in msg:
                print(
                    "ERROR: FashionCLIP failed to load due to incompatible transformers version.\n"
                    "Fix (recommended):\n"
                    "  pip install \"transformers<5\" \"huggingface-hub<1\" --upgrade --force-reinstall\n"
                    "Alternative (no model download):\n"
                    "  python -m scripts.encode_outfits_dummy\n"
                )
                return
            raise

        print("Model loaded")

        # Optional safety check: warn if DB has outfits not present in CSV.
        csv_path = os.path.abspath(args.csv_path)
        if os.path.exists(csv_path):
            try:
                csv_urls, csv_names, csv_rows = _read_csv_keys(csv_path)
                db_all = db.query(Outfit.id, Outfit.name, Outfit.shopee_url).all()
                extras = [
                    (oid, name, url)
                    for (oid, name, url) in db_all
                    if not _in_csv(name=name or "", url=url or "", csv_urls=csv_urls, csv_names=csv_names)
                ]
                if extras:
                    print(
                        f"WARNING: DB has {len(db_all)} outfits but CSV has {csv_rows} rows; "
                        f"{len(extras)} outfit(s) exist in DB but not in CSV: {csv_path}"
                    )
                    for item in extras[:10]:
                        print("  EXTRA", item)
                    if args.fail_on_mismatch:
                        print(
                            "ERROR: mismatch detected (--fail-on-mismatch enabled). "
                            "Aborting without encoding. "
                            "Fix by pruning extra DB rows or re-run without --fail-on-mismatch."
                        )
                        return
            except Exception as exc:
                print(f"WARNING: CSV cross-check failed: {exc}")

        # Lấy outfit cần encode
        query = db.query(Outfit)
        if not args.force:
            query = query.filter(Outfit.vector_blob.is_(None))
        outfits = query.all()

        if not outfits:
            if args.force:
                print("ℹ️  Không có outfit nào để encode.")
            else:
                print("ℹ️  Tất cả outfit đã có vector — không cần encode lại.")
            return

        print(f"Encoding {len(outfits)} outfits...")

        success_count = 0
        for i, outfit in enumerate(outfits, 1):
            print(f"  [{i}/{len(outfits)}] {outfit.name}...", end=" ")

            # Tải ảnh
            image = download_image(outfit.image_url)
            if image is None:
                print("SKIP")
                continue

            # Encode ảnh → vector 512 dims
            image_embedding = fclip.encode_images(
                [image], batch_size=1
            )  # shape: (1, 512)

            # Normalize vector (L2 norm) — cosine similarity = dot product
            vector = image_embedding[0].astype(np.float32)
            vector = vector / np.linalg.norm(vector)

            # Lưu BLOB vào DB
            outfit.vector_blob = vector.tobytes()
            success_count += 1
            print("OK")

        db.commit()
        print(f"\nOK Encode done: {success_count}/{len(outfits)} outfits.")

    finally:
        db.close()


if __name__ == "__main__":
    encode_all()
