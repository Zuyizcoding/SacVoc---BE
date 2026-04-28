"""
Schema cho màn "Xem lại kết quả" — History Viewer.

Tuân thủ: backend-python #3 — Pydantic models cho mọi IO.
"""
from pydantic import BaseModel

from app.schemas.color import ColorApiResponse
from app.schemas.body import BodyApiResponse


class HistoryApiResponse(BaseModel):
    """Response trả về lịch sử F1 + F2 (nếu có)."""
    color: ColorApiResponse | None = None
    body: BodyApiResponse | None = None
