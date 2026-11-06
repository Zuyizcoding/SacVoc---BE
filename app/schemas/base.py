"""Envelope JSON dùng chung cho các endpoint."""

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Wrapper chuẩn — frontend luôn check `success` trước."""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str | int] = None


class HealthResponse(BaseModel):
    """Response cho GET /health — dùng warm-up server trước demo."""

    status: str = "ok"
    version: str = "1.0.0"
