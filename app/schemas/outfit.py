"""
Pydantic schemas cho F3 — Gợi ý trang phục.
Tuân thủ: backend-python #3 — Không dùng dict trần.
"""
from pydantic import BaseModel, Field


class OutfitItem(BaseModel):
    """1 outfit trong danh sách gợi ý."""
    id: int
    name: str
    category: str
    image_url: str
    shopee_url: str
    price_range: str
    score: float = Field(
        description="Cosine similarity score 0.0-1.0"
    )


class OutfitRecommendResponse(BaseModel):
    """Response cho GET /api/v1/outfits/recommend — top 6 outfit."""
    query_used: str               # Text query gửi vào FashionCLIP
    outfits: list[OutfitItem]     # Top-6 outfit theo cosine similarity


class OutfitDetailResponse(BaseModel):
    """Response cho GET /api/v1/outfits/{id}."""
    id: int
    name: str
    category: str
    image_url: str
    shopee_url: str
    price_range: str
    explanation: str | None = None  # SHOULD: Gemini generate real-time
