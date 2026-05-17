"""
Pydantic schemas cho F1 — Phân tích màu sắc cá nhân.
Tuân thủ: backend-python #3 — Type Hints cho mọi field.

LƯU Ý: ColorGeminiResponse là schema truyền vào Gemini `response_schema`
        (Tính năng #11 — Structured Output). Gemini SẼ trả đúng format này.
        ColorApiResponse là schema trả về cho Flutter client.
"""
from pydantic import BaseModel, Field
from pydantic import model_validator


class MakeupGuide(BaseModel):
    """Hướng dẫn trang điểm theo sub-season."""
    lip: list[str] = Field(description="3 màu son phù hợp")
    eye: list[str] = Field(description="3 màu phấn mắt phù hợp")
    hair: list[str] = Field(description="2-3 màu tóc phù hợp")


class ColorGeminiResponse(BaseModel):
    """Schema ép Gemini trả đúng JSON (Tính năng #11).
    Truyền vào `generation_config.response_schema`.
    """
    # Edge-case: multiple people detected -> return a structured error.
    error_code: str | None = Field(
        default=None,
        description=(
            "Nếu phát hiện nhiều người trong ảnh: 'MULTIPLE_PEOPLE_DETECTED'. "
            "Nếu phân tích bình thường: null/không có."
        ),
    )
    message: str | None = Field(
        default=None,
        description="Thông báo tiếng Việt cho người dùng khi có error_code.",
    )

    # Normal successful analysis fields
    sub_season: str | None = Field(
        default=None,
        description=(
            "1 trong 12: Light Spring, True Spring, Bright Spring, Light Summer, "
            "True Summer, Soft Summer, Soft Autumn, True Autumn, Deep Autumn, "
            "Deep Winter, True Winter, Bright Winter"
        ),
    )
    primary_season: str | None = Field(
        default=None,
        description="Spring | Summer | Autumn | Winter",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mức tự tin 0.0-1.0",
    )
    secondary_season: str | None = Field(default=None, description="Sub-season thứ 2 có thể")
    explanation: str | None = Field(default=None, description="2-3 câu tiếng Việt giải thích lý do")
    palette: list[str] | None = Field(default=None, description="6 mã hex (#RRGGBB)")
    makeup: MakeupGuide | None = None
    advice: str | None = Field(default=None, description="1 câu lời khuyên ngắn gọn tiếng Việt")

    @model_validator(mode="after")
    def _validate_either_error_or_success(self):
        # Error path
        if self.error_code is not None:
            if not self.message:
                raise ValueError("message is required when error_code is set")
            return self

        # Success path: require all normal fields
        missing = []
        if not self.sub_season:
            missing.append("sub_season")
        if not self.primary_season:
            missing.append("primary_season")
        if self.confidence is None:
            missing.append("confidence")
        if not self.secondary_season:
            missing.append("secondary_season")
        if not self.explanation:
            missing.append("explanation")
        if not self.palette:
            missing.append("palette")
        if self.makeup is None:
            missing.append("makeup")
        if not self.advice:
            missing.append("advice")

        if missing:
            raise ValueError(f"Missing required fields for successful analysis: {', '.join(missing)}")

        return self


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
