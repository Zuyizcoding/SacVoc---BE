"""
Pydantic schemas cho F2 — Phân tích dáng người.
Tuân thủ: backend-python #3 — Pydantic models cho mọi IO.
"""
from pydantic import BaseModel, Field, model_validator


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
    # Edge-case: clothing too wide (puffy/bulky) -> return a structured error.
    error_code: str | None = Field(
        default=None,
        description=(
            "Nếu trang phục quá rộng che dáng: 'CLOTHING_TOO_WIDE'. "
            "Nếu phân tích bình thường: null/không có."
        ),
    )
    message: str | None = Field(
        default=None,
        description="Thông báo tiếng Việt cho người dùng khi có error_code.",
    )

    # Normal successful analysis fields
    shape: str | None = Field(
        default=None,
        description=(
            "1 trong 5: Hourglass | Soft Hourglass | Rectangle | "
            "Triangle | Inverted Triangle"
        ),
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    measurements: BodyMeasurements | None = None
    styling_tips: list[str] | None = Field(default=None, description="3 mẹo mặc đồ tiếng Việt")
    avoid: list[str] | None = Field(default=None, description="2 kiểu đồ nên tránh tiếng Việt")
    explanation: str | None = Field(default=None, description="2-3 câu tiếng Việt giải thích")

    @model_validator(mode="after")
    def _validate_either_error_or_success(self):
        # Error path
        if self.error_code is not None:
            if not self.message:
                raise ValueError("message is required when error_code is set")
            return self

        # Success path: require all normal fields
        missing = []
        if not self.shape:
            missing.append("shape")
        if self.confidence is None:
            missing.append("confidence")
        if self.measurements is None:
            missing.append("measurements")
        if not self.styling_tips:
            missing.append("styling_tips")
        if not self.avoid:
            missing.append("avoid")
        if not self.explanation:
            missing.append("explanation")

        if missing:
            raise ValueError(f"Missing required fields for successful analysis: {', '.join(missing)}")

        return self


class BodyApiResponse(BaseModel):
    """Response trả về cho Flutter."""
    shape: str
    confidence: float
    measurements: BodyMeasurements
    styling_tips: list[str]
    avoid: list[str]
    explanation: str
