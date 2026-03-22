"""
FashionCLIP Encoding — Chạy 1 LẦN trên máy dev.

Tính năng #29:
  1. Load FashionCLIP model (fashion-clip)
  2. Đọc 80 ảnh outfit từ URLs trong DB
  3. Encode mỗi ảnh → vector 512 dims (float32)
  4. Lưu vector dạng BLOB vào cột `vector_blob` trong bảng Outfit

Cách chạy:
    pip install fashion-clip Pillow requests
    cd src/backend
    python -m scripts.encode_outfits

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import requests
from io import BytesIO
from PIL import Image

from app.db.database import SessionLocal, init_db
from app.db.models import Outfit


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
        print("❌ Chưa cài fashion-clip. Chạy: pip install fashion-clip")
        return

    init_db()
    db = SessionLocal()

    try:
        # Khởi tạo model (lần đầu sẽ download ~400MB)
        print("🔄 Loading FashionCLIP model...")
        fclip = FashionCLIP("fashion-clip")
        print("✅ Model loaded!")

        # Lấy outfit chưa có vector
        outfits = db.query(Outfit).filter(
            Outfit.vector_blob.is_(None)
        ).all()

        if not outfits:
            print("ℹ️  Tất cả outfit đã có vector — không cần encode lại.")
            return

        print(f"📦 Encoding {len(outfits)} outfits...")

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
            print("OK ✓")

        db.commit()
        print(f"\n✅ Encode xong: {success_count}/{len(outfits)} outfits.")

    finally:
        db.close()


if __name__ == "__main__":
    encode_all()
