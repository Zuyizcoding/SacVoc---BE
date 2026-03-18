"""
Schema response chuẩn cho toàn bộ API.
Tuân thủ: backend-python #3 — Pydantic models cho mọi IO.
"""
from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Wrapper chuẩn — frontend luôn check `success` trước."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[int] = None


class HealthResponse(BaseModel):
    """Response cho GET /health — dùng warm-up server trước demo."""
    status: str = "ok"
    version: str = "1.0.0"
