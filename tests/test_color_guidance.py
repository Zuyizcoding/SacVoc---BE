from app.api.endpoints.analyze import (
    _apply_soft_quality_adjustments,
    _build_demo_body_payload,
    _build_demo_color_payload,
    _canonicalize_color_result,
    _finalize_color_result,
    _soft_quality_confidence_cap,
)
from app.data.season_data import get_menswear_recommendation


def test_menswear_guidance_varies_by_sub_season() -> None:
    light_summer = get_menswear_recommendation("Light Summer")
    true_summer = get_menswear_recommendation("True Summer")

    assert light_summer != true_summer
    assert "Xanh baby" in light_summer["tops"]
    assert "Xanh dusty" in true_summer["tops"]


def test_canonicalize_color_result_keeps_advice_separate_from_menswear() -> None:
    payload = {
        "sub_season": "Soft Autumn",
        "primary_season": "Autumn",
        "confidence": 0.82,
        "secondary_season": "True Autumn",
        "explanation": "Test explanation",
        "palette": ["#000000"] * 6,
        "makeup": {
            "lip": ["x"],
            "eye": ["y"],
            "hair": ["z"],
        },
        "advice": "Ưu tiên bảng màu ấm.",
    }

    result = _canonicalize_color_result(payload)

    assert result["advice"] == "Ưu tiên bảng màu ấm."
    assert result["menswear"]["tops"]


def test_finalize_color_result_fills_missing_optional_text_fields() -> None:
    result = _finalize_color_result(
        {
            "sub_season": "Soft Autumn",
            "confidence": 0.62,
        },
        {
            "soft_far_subject_for_color": False,
            "soft_low_resolution": False,
            "soft_low_detail": False,
            "soft_bad_lighting": False,
            "soft_color_cast": False,
            "photo_quality_score": 0.91,
            "photo_quality_band": "excellent",
        },
    )

    assert result["primary_season"] == "Autumn"
    assert result["secondary_season"] == "Soft Autumn"
    assert result["photo_quality_score"] == 0.91
    assert result["photo_quality_band"] == "excellent"
    assert result["explanation"]
    assert result["advice"]
    assert result["makeup"]["lip"]
    assert result["menswear"]["tops"]


def test_soft_quality_adjustments_cap_far_subject_confidence_more_aggressively() -> None:
    result = _apply_soft_quality_adjustments(
        {
            "confidence": 0.92,
            "explanation": "Test explanation",
            "advice": "Test advice",
        },
        {
            "soft_far_subject_for_color": True,
            "soft_low_resolution": False,
            "soft_low_detail": False,
            "soft_bad_lighting": False,
            "soft_color_cast": False,
            "photo_quality_score": 0.63,
            "photo_quality_band": "usable",
        },
    )

    assert result["confidence"] == 0.55
    assert "chủ thể trong ảnh hơi nhỏ hoặc hơi xa" in result["explanation"]


def test_soft_quality_confidence_cap_is_gentler_for_single_mild_issue() -> None:
    cap = _soft_quality_confidence_cap(
        {
            "soft_far_subject_for_color": False,
            "soft_low_resolution": False,
            "soft_low_detail": False,
            "soft_bad_lighting": True,
            "soft_color_cast": False,
            "photo_quality_score": 0.68,
            "photo_quality_band": "usable",
        }
    )

    assert cap == 0.7


def test_soft_quality_confidence_cap_gets_stronger_for_multiple_issues() -> None:
    cap = _soft_quality_confidence_cap(
        {
            "soft_far_subject_for_color": False,
            "soft_low_resolution": True,
            "soft_low_detail": True,
            "soft_bad_lighting": True,
            "soft_color_cast": False,
            "photo_quality_score": 0.46,
            "photo_quality_band": "poor",
        }
    )

    assert cap == 0.58


def test_build_demo_color_payload_attaches_quality_metadata_and_notice(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.endpoints.analyze.get_demo_color_result",
        lambda: {
            "sub_season": "Soft Autumn",
            "confidence": 0.72,
            "secondary_season": "True Autumn",
            "explanation": "Đây là kết quả mẫu.",
            "advice": "Ưu tiên bảng màu ấm.",
        },
    )

    result = _build_demo_color_payload(
        {
            "soft_far_subject_for_color": False,
            "soft_low_resolution": False,
            "soft_low_detail": False,
            "soft_bad_lighting": False,
            "soft_color_cast": False,
            "photo_quality_score": 0.63,
            "photo_quality_band": "usable",
        }
    )

    assert result["photo_quality_score"] == 0.63
    assert result["photo_quality_band"] == "usable"
    assert result["analysis_source"] == "degraded_demo"
    assert "kết quả dự phòng" in result["explanation"]
    assert "chế độ dự phòng" in result["advice"]


def test_build_demo_body_payload_attaches_quality_metadata_and_notice(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.endpoints.analyze.get_demo_body_result",
        lambda: {
            "shape": "Rectangle",
            "confidence": 0.75,
            "measurements": {
                "shoulder_relative": "balanced",
                "hip_relative": "balanced",
                "waist_definition": "moderate",
            },
            "styling_tips": ["Tip 1", "Tip 2", "Tip 3"],
            "avoid": ["Avoid 1", "Avoid 2"],
            "explanation": "Đây là kết quả mẫu.",
        },
    )

    result = _build_demo_body_payload(
        {
            "soft_far_subject_for_color": False,
            "soft_low_resolution": False,
            "soft_low_detail": False,
            "soft_bad_lighting": False,
            "soft_color_cast": False,
            "photo_quality_score": 0.68,
            "photo_quality_band": "usable",
        }
    )

    assert result["photo_quality_score"] == 0.68
    assert result["photo_quality_band"] == "usable"
    assert result["confidence"] == 0.75
    assert result["analysis_source"] == "degraded_demo"
    assert "kết quả dự phòng" in result["explanation"]
