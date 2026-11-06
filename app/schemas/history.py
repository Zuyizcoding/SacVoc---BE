"""Hai kết quả gần nhất, mỗi kết quả có thể chưa tồn tại."""

from pydantic import BaseModel

from app.schemas.body import BodyApiResponse
from app.schemas.color import ColorApiResponse


class HistoryApiResponse(BaseModel):
    """Response trả về lịch sử F1 + F2 (nếu có)."""

    color: ColorApiResponse | None = None
    body: BodyApiResponse | None = None
