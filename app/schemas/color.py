"""
Pydantic schemas cho F1 — Phân tích màu sắc cá nhân.
Tuân thủ: backend-python #3 — Type Hints cho mọi field.

LƯU Ý: ColorGeminiResponse là schema truyền vào Gemini `response_schema`
        (Tính năng #11 — Structured Output). Gemini SẼ trả đúng format này.
        ColorApiResponse là schema trả về cho Flutter client.
"""
from pydantic import BaseModel, Field


class MakeupGuide(BaseModel):
    """Hướng dẫn trang điểm theo sub-season."""
    lip: list[str] = Field(description="3 màu son phù hợp")
    eye: list[str] = Field(description="3 màu phấn mắt phù hợp")
    hair: list[str] = Field(description="2-3 màu tóc phù hợp")


class ColorGeminiResponse(BaseModel):
    """Schema ép Gemini trả đúng JSON (Tính năng #11).
    Truyền vào `generation_config.response_schema`.
    """
    sub_season: str = Field(
        description="1 trong 12: Light Spring, True Spring, Bright Spring, "
        "Light Summer, True Summer, Soft Summer, Soft Autumn, True Autumn, "
        "Deep Autumn, Deep Winter, True Winter, Bright Winter"
    )
    primary_season: str = Field(
        description="Spring | Summer | Autumn | Winter"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Mức tự tin 0.0-1.0"
    )
    secondary_season: str = Field(
        description="Sub-season thứ 2 có thể"
    )
    explanation: str = Field(
        description="2-3 câu tiếng Việt giải thích lý do"
    )
    palette: list[str] = Field(
        description="6 mã hex (#RRGGBB)"
    )
    makeup: MakeupGuide
    advice: str = Field(
        description="1 câu lời khuyên ngắn gọn tiếng Việt"
    )


class ColorApiResponse(BaseModel):
    """Response trả về cho Flutter — sau khi override palette từ season_data.py."""
    sub_season: str
    primary_season: str
    confidence: float
    secondary_season: str
    explanation: str
    palette: list[str]       # 6 hex — override từ season_data.py (Tính năng #20)
    makeup: MakeupGuide
    advice: str
