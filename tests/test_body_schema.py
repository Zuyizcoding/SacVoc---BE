from app.schemas.body import BodyGeminiResponse


def test_body_response_does_not_autofill_measurements_from_shape_only() -> None:
    payload = {
        "shape": "Rectangle",
        "confidence": 0.41,
        "styling_tips": ["Tip 1", "Tip 2", "Tip 3"],
        "avoid": ["Avoid 1", "Avoid 2"],
        "explanation": "Kết quả tham khảo từ ảnh.",
    }
    result = BodyGeminiResponse.model_validate(payload)
    assert result.shape == "Rectangle"
    assert result.measurements is None
    assert "measurements" in result.missing_detail_fields()


def test_body_response_still_coerces_string_measurements_when_present() -> None:
    payload = {
        "shape": "Triangle",
        "confidence": 0.44,
        "measurements": "Vai hẹp, hông rộng, eo rõ",
        "styling_tips": ["Tip 1", "Tip 2", "Tip 3"],
        "avoid": ["Avoid 1", "Avoid 2"],
        "explanation": "Kết quả tham khảo từ ảnh.",
    }
    result = BodyGeminiResponse.model_validate(payload)
    assert result.measurements is not None
    assert result.measurements.shoulder_relative == "narrow"
    assert result.measurements.hip_relative == "broad"
