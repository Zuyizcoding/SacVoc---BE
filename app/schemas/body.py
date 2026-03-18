"""
Pydantic schemas cho F2 — Phân tích dáng người.
Tuân thủ: backend-python #3 — Pydantic models cho mọi IO.
"""
from pydantic import BaseModel, Field


class BodyMeasurements(BaseModel):
    """Tỷ lệ tương đối — Gemini đánh giá từ ảnh, không phải đo pixel."""
    shoulder_relative: str = Field(
        description="broad | balanced | narrow"
    )
    hip_relative: str = Field(
        description="broad | balanced | narrow"
    )
    waist_definition: str = Field(
        description="defined | moderate | minimal"
    )


class BodyGeminiResponse(BaseModel):
    """Schema ép Gemini trả đúng JSON (Tính năng #11).
    Truyền vào `generation_config.response_schema`.
    """
    shape: str = Field(
        description="1 trong 5: Hourglass | Soft Hourglass | "
        "Rectangle | Triangle | Inverted Triangle"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    measurements: BodyMeasurements
    styling_tips: list[str] = Field(
        description="3 mẹo mặc đồ tiếng Việt"
    )
    avoid: list[str] = Field(
        description="2 kiểu đồ nên tránh tiếng Việt"
    )
    explanation: str = Field(
        description="2-3 câu tiếng Việt giải thích"
    )


class BodyApiResponse(BaseModel):
    """Response trả về cho Flutter."""
    shape: str
    confidence: float
    measurements: BodyMeasurements
    styling_tips: list[str]
    avoid: list[str]
    explanation: str
