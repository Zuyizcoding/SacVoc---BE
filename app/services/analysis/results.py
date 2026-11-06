"""Chuẩn hóa kết quả AI và dựng kết quả demo trước khi lưu."""

from app.data.season_data import (
    canonicalize_sub_season,
    get_menswear_recommendation,
    get_season_entry,
)
from app.schemas.body import BodyGeminiResponse
from app.services.demo_cache import get_demo_body_result, get_demo_color_result

from .photo_quality import (
    _apply_soft_quality_adjustments,
    _photo_quality_score_value,
    _soft_quality_confidence_cap,
)
from .photo_types import PhotoQualityReport

DEMO_CACHE_COLOR_NOTICE = "Lưu ý: hệ thống đang dùng kết quả dự phòng vì AI tạm thời quá tải, nên đây là gợi ý demo tham khảo."
DEMO_CACHE_BODY_NOTICE = DEMO_CACHE_COLOR_NOTICE


def _append_notice(text: str, notice: str) -> str:
    base = text.strip()
    if not base:
        return notice
    if notice in base:
        return base
    return f"{base} {notice}"


def _canonicalize_color_result(result: dict) -> dict:
    """Chuẩn hóa F1 bằng bảng season_data trước khi lưu/trả về."""
    sub_season_raw = str(result.get("sub_season") or "").strip()
    canonical_sub_season = canonicalize_sub_season(sub_season_raw)
    if canonical_sub_season is None:
        raise ValueError(f"Invalid sub-season from Gemini: {sub_season_raw}")
    entry = get_season_entry(canonical_sub_season)
    result["sub_season"] = canonical_sub_season
    result["primary_season"] = entry["primary"]
    result["palette"] = entry["palette"]
    result["makeup"] = entry["makeup"]
    result["menswear"] = get_menswear_recommendation(canonical_sub_season)
    return result


def merge_nonempty_result(base: dict, patch: dict) -> dict:
    """Merge partial F2 payloads, ưu tiên dữ liệu mới khi field có giá trị."""
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            continue
        if isinstance(value, list) and (not value):
            continue
        if isinstance(value, str) and (not value.strip()):
            continue
        merged[key] = value
    return merged


def _color_missing_fields(result: dict) -> list[str]:
    missing: list[str] = []
    sub_season = str(result.get("sub_season") or "").strip()
    if not sub_season or canonicalize_sub_season(sub_season) is None:
        missing.append("sub_season")
    confidence_raw = result.get("confidence")
    if not isinstance(confidence_raw, (int, float)):
        missing.append("confidence")
    if not str(result.get("explanation") or "").strip():
        missing.append("explanation")
    if not str(result.get("advice") or "").strip():
        missing.append("advice")
    return missing


def _fallback_color_explanation(
    sub_season: str, quality_flags: PhotoQualityReport
) -> str:
    explanation = f"Hệ thống nhận diện bạn gần nhất với nhóm {sub_season} dựa trên tổng thể da, tóc và mắt trong ảnh."
    if quality_flags.soft_far_subject_for_color or quality_flags.soft_low_detail:
        return f"{explanation} Tuy vậy ảnh hiện tại chưa thật sự lý tưởng nên kết quả này nên được dùng như tham khảo."
    return explanation


def _fallback_color_advice(result: dict) -> str:
    menswear = result.get("menswear")
    if isinstance(menswear, dict):
        tops = [
            item
            for item in menswear.get("tops", [])
            if isinstance(item, str) and item.strip()
        ]
        bottoms = [
            item
            for item in menswear.get("bottoms", [])
            if isinstance(item, str) and item.strip()
        ]
        if tops and bottoms:
            return f"Ưu tiên {tops[0]} cho áo/layer và phối cùng {bottoms[0]} để tổng thể dễ hài hòa hơn."
        if tops:
            return f"Ưu tiên các gam như {tops[0]} cho áo hoặc layer để gương mặt trông sáng và cân bằng hơn."
    return "Ưu tiên các gam màu nằm trong bảng gợi ý và chọn outfit có độ tương phản vừa phải để tôn da hơn."


def _finalize_color_result(result: dict, quality_flags: PhotoQualityReport) -> dict:
    patched = _canonicalize_color_result(dict(result))
    confidence_cap = _soft_quality_confidence_cap(quality_flags)
    has_soft_warning = confidence_cap is not None
    photo_quality_score = _photo_quality_score_value(quality_flags)
    confidence_raw = patched.get("confidence")
    if isinstance(confidence_raw, (int, float)):
        patched["confidence"] = float(confidence_raw)
    else:
        baseline = 0.48 + 0.34 * photo_quality_score
        patched["confidence"] = min(0.84, round(baseline, 4))
        if has_soft_warning:
            patched["confidence"] = min(patched["confidence"], 0.7)
    if not str(patched.get("secondary_season") or "").strip():
        patched["secondary_season"] = patched["sub_season"]
    if not str(patched.get("explanation") or "").strip():
        patched["explanation"] = _fallback_color_explanation(
            patched["sub_season"], quality_flags
        )
    if not str(patched.get("advice") or "").strip():
        patched["advice"] = _fallback_color_advice(patched)
    return _apply_soft_quality_adjustments(patched, quality_flags)


def _build_demo_color_payload(quality_flags: PhotoQualityReport) -> dict:
    payload = _finalize_color_result(get_demo_color_result(), quality_flags)
    payload["analysis_source"] = "degraded_demo"
    payload["explanation"] = _append_notice(
        str(payload.get("explanation") or "").strip(), DEMO_CACHE_COLOR_NOTICE
    )
    payload["advice"] = _append_notice(
        str(payload.get("advice") or "").strip(),
        "Lưu ý: hệ thống đang dùng chế độ dự phòng do AI tạm thời quá tải.",
    )
    return payload


def _build_demo_body_payload(quality_flags: PhotoQualityReport) -> dict:
    payload = BodyGeminiResponse.model_validate(get_demo_body_result()).model_dump()
    payload = _apply_soft_quality_adjustments(payload, quality_flags)
    payload["analysis_source"] = "degraded_demo"
    if payload.get("confidence") is None:
        payload["confidence"] = round(
            min(0.72, 0.38 + 0.28 * _photo_quality_score_value(quality_flags)), 4
        )
    payload["explanation"] = _append_notice(
        str(payload.get("explanation") or "").strip(), DEMO_CACHE_BODY_NOTICE
    )
    return payload
