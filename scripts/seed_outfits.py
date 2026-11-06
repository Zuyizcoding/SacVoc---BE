"""Seed placeholder cũ; catalog hiện hành được nhập bằng seed_outfits_csv."""

import os
import sys

# Thêm src/backend vào path để import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import SessionLocal, init_db
from app.db.models import Outfit

# ═══════════════════════════════════════════════
# DỮ LIỆU MẪU — Team thay thế bằng data thật
# ═══════════════════════════════════════════════

SAMPLE_OUTFITS: list[dict] = [
    # ─── DAILY (30 items) ─────────────────────
    {
        "name": "Áo thun trắng basic + Quần jeans xanh",
        "category": "set",
        "image_url": "https://placeholder.com/outfit_001.jpg",
        "shopee_url": "https://shopee.vn/product/001",
        "price_range": "200k-350k",
        "occasion": "daily",
    },
    {
        "name": "Váy midi hoa nhí pastel",
        "category": "dress",
        "image_url": "https://placeholder.com/outfit_002.jpg",
        "shopee_url": "https://shopee.vn/product/002",
        "price_range": "300k-500k",
        "occasion": "daily",
    },
    {
        "name": "Áo sơ mi oversize + Quần culottes",
        "category": "set",
        "image_url": "https://placeholder.com/outfit_003.jpg",
        "shopee_url": "https://shopee.vn/product/003",
        "price_range": "350k-550k",
        "occasion": "daily",
    },
    # ... Team thêm 27 items daily nữa ...
    # ─── WORK (20 items) ──────────────────────
    {
        "name": "Áo blazer trắng + Quần âu đen",
        "category": "set",
        "image_url": "https://placeholder.com/outfit_021.jpg",
        "shopee_url": "https://shopee.vn/product/021",
        "price_range": "500k-800k",
        "occasion": "work",
    },
    {
        "name": "Váy liền công sở xanh navy",
        "category": "dress",
        "image_url": "https://placeholder.com/outfit_022.jpg",
        "shopee_url": "https://shopee.vn/product/022",
        "price_range": "400k-650k",
        "occasion": "work",
    },
    # ... Team thêm 18 items work nữa ...
    # ─── PARTY (15 items) ─────────────────────
    {
        "name": "Váy sequin đen ngắn",
        "category": "dress",
        "image_url": "https://placeholder.com/outfit_041.jpg",
        "shopee_url": "https://shopee.vn/product/041",
        "price_range": "600k-1.2tr",
        "occasion": "party",
    },
    # ... Team thêm 14 items party nữa ...
    # ─── DATE (15 items) ──────────────────────
    {
        "name": "Áo trễ vai + Chân váy xếp ly",
        "category": "set",
        "image_url": "https://placeholder.com/outfit_056.jpg",
        "shopee_url": "https://shopee.vn/product/056",
        "price_range": "350k-600k",
        "occasion": "date",
    },
    # ... Team thêm 14 items date nữa ...
]


def seed() -> None:
    """Nhập outfit vào DB — kiểm tra trùng trước khi insert."""
    init_db()
    db = SessionLocal()

    try:
        existing_count = db.query(Outfit).count()
        if existing_count > 0:
            print(f"⚠️  DB đã có {existing_count} outfits — bỏ qua seed.")
            print("   Nếu muốn seed lại, xóa file DB trước.")
            return

        for item in SAMPLE_OUTFITS:
            outfit = Outfit(**item)
            db.add(outfit)

        db.commit()
        print(f"✅ Đã nhập {len(SAMPLE_OUTFITS)} outfits vào DB.")
        print("   → Chạy encode_outfits.py tiếp để tạo vector embeddings.")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
