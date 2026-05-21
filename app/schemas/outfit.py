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
        description="Final reranked score 0.0-1.0"
    )
    retrieval_score: float | None = Field(
        default=None,
        description="Raw semantic similarity before reranking"
    )
    reason_tags: list[str] | None = Field(
        default=None,
        description="Short rule tags explaining why this outfit ranked well"
    )


class OutfitRecommendResponse(BaseModel):
    """Response cho GET /api/v1/outfits/recommend — top 6 outfit."""
    query_used: str               # Text query gửi vào FashionCLIP
    query_source: str | None = None
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
