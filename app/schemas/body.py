"""Pydantic schemas cho F2 — Phân tích dáng người.

Tuân thủ: backend-python #3 — Pydantic models cho mọi IO.
"""
import re

from pydantic import BaseModel, Field, model_validator


def _split_to_list(value: object, *, max_items: int) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        return items[:max_items] if items else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        # Common bullet formats: lines, '-', '•', ';'
        raw_parts: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = line.lstrip("-•*\t ").strip()
            if line:
                raw_parts.append(line)

        if not raw_parts:
            raw_parts = [p.strip() for p in re.split(r"[;]+", text) if p.strip()]  # type: ignore[name-defined]

        return raw_parts[:max_items] if raw_parts else None

    # Fallback: coerce to string
    s = str(value).strip()
    return [s] if s else None


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

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_keys_and_fill_defaults(cls, data):
        """Gemini đôi khi trả key khác schema (vd: `recommend`).

        Với F2 MVP, ưu tiên: không fail toàn bộ request nếu đã có `shape`.
        """
        if not isinstance(data, dict):
            return data

        # If it's an error response, don't force success fields.
        if data.get("error_code") is not None:
            return data

        # Legacy/synonym keys → schema keys
        if not data.get("shape"):
            for k in ("body_shape", "result", "type"):
                if isinstance(data.get(k), str) and str(data[k]).strip():
                    data["shape"] = str(data[k]).strip()
                    break

        if data.get("styling_tips") is None:
            for k in ("stylingTips", "tips", "advice", "recommend", "recommendations"):
                if k in data and data.get(k) is not None:
                    data["styling_tips"] = data.get(k)
                    break

        # Coerce list-ish fields
        data["styling_tips"] = _split_to_list(data.get("styling_tips"), max_items=3)
        data["avoid"] = _split_to_list(data.get("avoid"), max_items=2)

        # If Gemini didn't provide measurements, fill a neutral default.
        if data.get("measurements") is None:
            data["measurements"] = {
                "shoulder_relative": "balanced",
                "hip_relative": "balanced",
                "waist_definition": "moderate",
            }

        # Confidence default: conservative but non-blocking.
        if data.get("confidence") is None:
            data["confidence"] = 0.6

        # Explanation default if missing but we have a shape.
        if not data.get("explanation") and data.get("shape"):
            data["explanation"] = (
                "Kết quả được suy ra từ tỷ lệ vai–eo–hông quan sát được trong ảnh. "
                "Nếu ảnh mặc đồ quá rộng hoặc góc chụp lệch, độ chính xác có thể giảm."
            )

        # Avoid default if still missing
        if not data.get("avoid"):
            data["avoid"] = [
                "Đồ oversize quá rộng có thể che tỷ lệ cơ thể — ưu tiên phom vừa vặn",
                "Trang phục quá nhiều chi tiết ở phần bạn muốn thu gọn có thể làm tổng thể nặng nề",
            ]

        # Styling tips default if still missing
        if not data.get("styling_tips"):
            data["styling_tips"] = [
                "Ưu tiên trang phục có phom vừa vặn (tailored) để tôn tỷ lệ",
                "Dùng thắt lưng/chiết eo nhẹ để tạo điểm nhấn cân đối",
                "Chọn quần/váy dáng cơ bản (ống đứng/chữ A) để tổng thể gọn gàng",
            ]

        return data

    @model_validator(mode="after")
    def _validate_either_error_or_success(self):
        # Error path
        if self.error_code is not None:
            if not self.message:
                raise ValueError("message is required when error_code is set")
            return self

        # Success path: minimally require `shape`. Other fields can be defaulted
        # (Gemini sometimes omits them even with response_schema).
        if not self.shape:
            raise ValueError("Missing required field for successful analysis: shape")

        # Defensive defaults (should already be set by the `before` validator).
        if self.confidence is None:
            self.confidence = 0.6
        if self.measurements is None:
            self.measurements = BodyMeasurements(
                shoulder_relative="balanced",
                hip_relative="balanced",
                waist_definition="moderate",
            )
        if not self.styling_tips:
            self.styling_tips = [
                "Ưu tiên trang phục có phom vừa vặn (tailored) để tôn tỷ lệ",
                "Dùng thắt lưng/chiết eo nhẹ để tạo điểm nhấn cân đối",
                "Chọn quần/váy dáng cơ bản (ống đứng/chữ A) để tổng thể gọn gàng",
            ]
        if not self.avoid:
            self.avoid = [
                "Đồ oversize quá rộng có thể che tỷ lệ cơ thể — ưu tiên phom vừa vặn",
                "Trang phục quá nhiều chi tiết ở phần bạn muốn thu gọn có thể làm tổng thể nặng nề",
            ]
        if not self.explanation:
            self.explanation = (
                "Kết quả được suy ra từ tỷ lệ vai–eo–hông quan sát được trong ảnh. "
                "Nếu ảnh mặc đồ quá rộng hoặc góc chụp lệch, độ chính xác có thể giảm."
            )

        return self


class BodyApiResponse(BaseModel):
    """Response trả về cho Flutter."""
    shape: str
    confidence: float
    measurements: BodyMeasurements
    styling_tips: list[str]
    avoid: list[str]
    explanation: str
