"""
SQLAlchemy ORM Models — Bảng cơ sở dữ liệu cho Sắc Vóc.

Tuân thủ:
  - backend-python #3: Models (SQLAlchemy) tách biệt Schemas (Pydantic)
  - Tính năng #7: SQLite persistence
  - Tính năng #15: AnalysisHistory lưu F1+F2 → phục vụ F3 combined flow
  - Tính năng #28: Outfit table — 80 items + vector BLOB
  - Tính năng #29: vector_blob lưu FashionCLIP embedding (encode offline)

QUAN HỆ BẢNG:
  ┌──────────────────┐
  │  AnalysisHistory  │  ← Lưu kết quả F1 (color) + F2 (body) theo device_id
  │  device_id (PK)  │
  └──────────────────┘
           │
           │ F3 đọc sub_season + shape từ đây
           ▼
  ┌──────────────────┐
  │      Outfit       │  ← 80 items, có vector_blob (FashionCLIP embedding)
  │    id (PK, AI)   │
  └──────────────────┘
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
)

from app.db.database import Base


class AnalysisHistory(Base):
    """Lưu kết quả phân tích F1 (Color) + F2 (Body) theo thiết bị.

    Tính năng #3: device_id = SHA256 từ device_info_plus (anonymous identity)
    Tính năng #15: F3 đọc sub_season + shape từ bảng này để build query

    Mỗi device_id chỉ có 1 row — UPDATE khi phân tích lại (không tạo mới).
    Lý do: MVP không cần lịch sử nhiều lần — chỉ cần kết quả gần nhất.
    """

    __tablename__ = "analysis_history"

    # ─── Primary Key: SHA256 device ID ───
    device_id: str = Column(
        String(64), primary_key=True, index=True,
        comment="SHA256 hash từ device_info_plus — anonymous identity"
    )

    # ─── F1 Color Results ───
    color_sub_season: str | None = Column(
        String(30), nullable=True,
        comment="VD: 'Soft Autumn', 'True Winter' — NULL nếu chưa phân tích F1"
    )
    color_primary_season: str | None = Column(
        String(10), nullable=True,
        comment="Spring | Summer | Autumn | Winter"
    )
    color_confidence: float | None = Column(
        Float, nullable=True,
        comment="0.0 → 1.0 — mức tự tin Gemini"
    )
    color_result_json: str | None = Column(
        Text, nullable=True,
        comment="Toàn bộ JSON response F1 — backup đầy đủ cho màn 'Xem lại'"
    )

    # ─── F2 Body Results ───
    body_shape: str | None = Column(
        String(30), nullable=True,
        comment="VD: 'Hourglass', 'Triangle' — NULL nếu chưa phân tích F2"
    )
    body_confidence: float | None = Column(
        Float, nullable=True,
        comment="0.0 → 1.0"
    )
    body_result_json: str | None = Column(
        Text, nullable=True,
        comment="Toàn bộ JSON response F2 — backup đầy đủ"
    )

    # ─── Timestamps ───
    color_analyzed_at: datetime | None = Column(
        DateTime(timezone=True), nullable=True,
        comment="Thời điểm phân tích F1 gần nhất"
    )
    body_analyzed_at: datetime | None = Column(
        DateTime(timezone=True), nullable=True,
        comment="Thời điểm phân tích F2 gần nhất"
    )
    created_at: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        comment="Lần đầu thiết bị gửi request"
    )
    updated_at: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Lần cuối cập nhật bất kỳ field nào"
    )


class Outfit(Base):
    """80 items trang phục — nhập tay bởi team (Tính năng #28).

    Tính năng #29: vector_blob = FashionCLIP image embedding (512 dims)
      - Encode 1 lần trên máy dev bằng `encode_outfits.py`
      - Server KHÔNG load FashionCLIP model — chỉ đọc vector từ DB
      - Vector lưu dạng BLOB (numpy tobytes) — nhẹ hơn JSON array

    Tính năng #30: Semantic Search tính cosine similarity giữa
      text query vector và image vector → top-6 outfit

    Phân bố occasion (SHOULD — Outfit Diversity):
      - daily: 30 items
      - work: 20 items
      - party: 15 items
      - date: 15 items
    """

    __tablename__ = "outfits"

    # ─── Primary Key ───
    id: int = Column(
        Integer, primary_key=True, autoincrement=True,
    )

    # ─── Thông tin outfit (nhập tay — Tính năng #28) ───
    name: str = Column(
        String(100), nullable=False,
        comment="VD: 'Váy wrap dress hoa nhí', 'Áo blazer trắng oversize'"
    )
    category: str = Column(
        String(50), nullable=False,
        comment="VD: 'dress', 'top', 'bottom', 'set', 'outerwear'"
    )
    image_url: str = Column(
        String(500), nullable=False,
        comment="URL ảnh outfit — host trên Cloudinary hoặc Supabase Storage"
    )
    shopee_url: str = Column(
        String(500), nullable=False, default="",
        comment="Link sản phẩm Shopee — Tính năng #33"
    )
    price_range: str = Column(
        String(50), nullable=False, default="",
        comment="VD: '200k-400k', '500k-800k'"
    )
    occasion: str = Column(
        String(20), nullable=False, default="daily",
        comment="daily | work | party | date — Tính năng #28"
    )

    gender: str = Column(
        String(10), nullable=False, default="any",
        comment="any | male | female — dùng để filter recommendation khi seed data có cả nam/nữ"
    )

    # ─── FashionCLIP Embedding (Tính năng #29) ───
    vector_blob: bytes | None = Column(
        LargeBinary, nullable=True,
        comment="FashionCLIP image embedding — 512 dims × float32 = 2048 bytes"
    )

    # ─── Metadata ───
    created_at: datetime = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
