"""Schema suy luận F1 và kết quả được enrich từ bảng mùa sắc."""

from pydantic import BaseModel, Field, model_validator


class MakeupGuide(BaseModel):
    """Hướng dẫn trang điểm theo sub-season."""

    lip: list[str] = Field(description="3 màu son phù hợp")
    eye: list[str] = Field(description="3 màu phấn mắt phù hợp")
    hair: list[str] = Field(description="2-3 màu tóc phù hợp")


class MenswearGuide(BaseModel):
    """Quick styling directions for male-focused color output."""

    tops: list[str] = Field(description="3 nhóm màu áo hoặc layer nên ưu tiên")
    bottoms: list[str] = Field(description="3 nhóm màu quần nên phối dễ nhất")
    accessories: list[str] = Field(description="2-3 gợi ý phụ kiện/kim loại phù hợp")


class ColorInferenceResponse(BaseModel):
    """Schema mỏng cho Gemini: chỉ giữ các field cốt lõi cần suy luận."""

    # Edge-case: multiple people detected -> return a structured error.
    error_code: str | None = Field(
        default=None,
        description=(
            "Nếu phát hiện nhiều người hoặc không có người trong ảnh: "
            "'MULTIPLE_PEOPLE_DETECTED' / 'NO_PERSON_DETECTED'. "
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
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mức tự tin 0.0-1.0",
    )
    secondary_season: str | None = Field(
        default=None,
        description="Sub-season thứ 2 có thể, nếu không chắc có thể bỏ trống",
    )
    explanation: str | None = Field(
        default=None, description="2-3 câu tiếng Việt giải thích lý do"
    )
    advice: str | None = Field(
        default=None,
        description="1-2 câu lời khuyên ngắn gọn tiếng Việt, ưu tiên trang phục nam",
    )
    primary_season: str | None = Field(
        default=None,
        description="Legacy optional field - backend có thể tự tính từ sub_season",
    )
    palette: list[str] | None = Field(
        default=None,
        description="Legacy optional field - backend sẽ override nếu Gemini trả thiếu",
    )
    makeup: MakeupGuide | None = Field(
        default=None,
        description="Legacy optional field - backend sẽ override từ season_data",
    )
    menswear: MenswearGuide | None = Field(
        default=None,
        description="Legacy optional field - backend sẽ override từ season_data",
    )

    @model_validator(mode="after")
    def _validate_either_error_or_success(self):
        # Error path
        if self.error_code is not None:
            if not self.message:
                raise ValueError("message is required when error_code is set")
            return self

        # Success path: require only the irreducible field from Gemini.
        if not self.sub_season:
            raise ValueError(
                "Missing required fields for successful analysis: sub_season"
            )

        return self


class ColorGeminiResponse(ColorInferenceResponse):
    """Backward-compatible alias for older imports/tests."""


class ColorApiResponse(BaseModel):
    """Response trả về cho Flutter — sau khi override palette từ season_data.py."""

    sub_season: str
    primary_season: str
    confidence: float
    analysis_source: str = "live"
    photo_quality_score: float | None = None
    photo_quality_band: str | None = None
    secondary_season: str
    explanation: str
    palette: list[str]  # 6 hex — override từ season_data.py (Tính năng #20)
    makeup: MakeupGuide
    advice: str
    menswear: MenswearGuide | None = None
