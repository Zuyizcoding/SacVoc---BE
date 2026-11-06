"""Giữ nguyên chính sách confidence và cảnh báo của F1/F2."""

from .photo_types import PhotoQualityReport


def _apply_clothing_soft_fail_adjustments(result: dict) -> dict:
    """Cap confidence and append warning when analysis is forced from wide clothing."""
    patched = dict(result)
    warning = (
        "Trang phục trong ảnh hơi rộng nên hệ thống phân tích theo hướng tham khảo. "
        "Bạn có thể dùng ảnh mặc đồ ôm sát hơn để tăng độ chính xác."
    )

    confidence_raw = patched.get("confidence")
    if isinstance(confidence_raw, (int, float)):
        patched["confidence"] = min(float(confidence_raw), 0.45)
    else:
        patched["confidence"] = 0.4

    explanation = str(patched.get("explanation") or "").strip()
    if explanation:
        if "Lưu ý:" not in explanation:
            patched["explanation"] = f"{explanation} Lưu ý: {warning}"
    else:
        patched["explanation"] = f"Lưu ý: {warning}"

    return patched


def needs_color_quality_hint(quality_flags: PhotoQualityReport) -> bool:
    return bool(quality_flags.soft_far_subject_for_color)


def needs_body_quality_hint(quality_flags: PhotoQualityReport) -> bool:
    return (
        bool(quality_flags.soft_low_detail)
        and not bool(quality_flags.soft_low_resolution)
        and not bool(quality_flags.soft_bad_lighting)
        and not bool(quality_flags.soft_color_cast)
    )


def _soft_quality_warning_message(quality_flags: PhotoQualityReport) -> str:
    reasons: list[str] = []
    if quality_flags.soft_far_subject_for_color:
        reasons.append("chủ thể trong ảnh hơi nhỏ hoặc hơi xa")
    if quality_flags.soft_low_resolution:
        reasons.append("độ phân giải còn thấp")
    if quality_flags.soft_low_detail:
        reasons.append("ảnh thiếu chi tiết")
    if quality_flags.soft_bad_lighting:
        reasons.append("ánh sáng chưa ổn định")
    if quality_flags.soft_color_cast:
        reasons.append("ảnh có lệch màu")

    joined = ", ".join(reasons) if reasons else "chất lượng ảnh chưa tối ưu"
    return (
        f"Ảnh hiện tại có {joined}, nên kết quả chỉ nên dùng để tham khảo. "
        "Bạn có thể chụp ảnh sáng hơn, cân bằng màu hơn và rõ hơn để tăng độ chính xác."
    )


def _photo_quality_score_value(quality_flags: PhotoQualityReport) -> float:
    raw_score = quality_flags.photo_quality_score
    if isinstance(raw_score, (int, float)):
        return max(0.0, min(float(raw_score), 1.0))
    return 0.75


def _photo_quality_band_value(quality_flags: PhotoQualityReport) -> str:
    raw_band = str(quality_flags.photo_quality_band or "").strip().lower()
    if raw_band in {"excellent", "good", "usable", "poor"}:
        return raw_band
    return classify_photo_quality(_photo_quality_score_value(quality_flags))


def _soft_quality_confidence_cap(quality_flags: PhotoQualityReport) -> float | None:
    far_subject = bool(quality_flags.soft_far_subject_for_color)
    low_resolution = bool(quality_flags.soft_low_resolution)
    low_detail = bool(quality_flags.soft_low_detail)
    bad_lighting = bool(quality_flags.soft_bad_lighting)
    color_cast = bool(quality_flags.soft_color_cast)
    photo_quality_score = _photo_quality_score_value(quality_flags)
    photo_quality_band = _photo_quality_band_value(quality_flags)

    if not (far_subject or low_resolution or low_detail or bad_lighting or color_cast):
        return None

    if photo_quality_band == "excellent":
        cap = 0.84
    elif photo_quality_band == "good":
        cap = 0.78
    elif photo_quality_band == "usable":
        cap = 0.70
    else:
        cap = 0.60

    if photo_quality_score < 0.52:
        cap = min(cap, 0.62)
    if photo_quality_score < 0.45:
        cap = min(cap, 0.56)
    if far_subject:
        cap = min(cap, 0.55)
    if low_resolution and photo_quality_score < 0.65:
        cap = min(cap, 0.66)
    if low_detail and photo_quality_score < 0.72:
        cap = min(cap, 0.72)
    if bad_lighting and photo_quality_score < 0.72:
        cap = min(cap, 0.72)
    if color_cast and photo_quality_score < 0.74:
        cap = min(cap, 0.74)

    soft_issue_count = sum(
        1 for flag in (low_resolution, low_detail, bad_lighting, color_cast) if flag
    )
    if soft_issue_count >= 2:
        cap = min(cap, 0.64)
    if soft_issue_count >= 3:
        cap = min(cap, 0.58)
    if far_subject and soft_issue_count >= 1:
        cap = min(cap, 0.52)

    return cap


def _attach_photo_quality_metadata(
    result: dict, quality_flags: PhotoQualityReport
) -> dict:
    patched = dict(result)
    patched["photo_quality_score"] = round(_photo_quality_score_value(quality_flags), 4)
    patched["photo_quality_band"] = _photo_quality_band_value(quality_flags)
    return patched


def _apply_soft_quality_adjustments(
    result: dict, quality_flags: PhotoQualityReport
) -> dict:
    """Lower confidence and add a friendly warning when image quality is suboptimal."""
    confidence_cap = _soft_quality_confidence_cap(quality_flags)
    patched = _attach_photo_quality_metadata(result, quality_flags)
    if confidence_cap is None:
        return patched

    warning = _soft_quality_warning_message(quality_flags)

    confidence_raw = patched.get("confidence")
    if isinstance(confidence_raw, (int, float)):
        patched["confidence"] = min(float(confidence_raw), confidence_cap)

    explanation = str(patched.get("explanation") or "").strip()
    if explanation:
        patched["explanation"] = f"{explanation} Lưu ý: {warning}"

    advice = patched.get("advice")
    if isinstance(advice, str) and advice.strip():
        patched["advice"] = f"{advice.strip()} Lưu ý: {warning}"

    return patched


def _color_close_up_guidance_message(_quality_flags: PhotoQualityReport) -> str:
    return (
        "Ảnh hiện tại chụp hơi xa nên phù hợp để phân tích dáng người hơn là phân tích màu sắc. "
        "Để phân tích màu chính xác hơn, hãy dùng ảnh selfie gần hơn, thấy rõ khuôn mặt và vùng cổ, "
        "ánh sáng đều và không đeo kính râm/phụ kiện che mặt."
    )


def classify_photo_quality(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    if score >= 0.72:
        return "good"
    if score >= 0.58:
        return "usable"
    return "poor"
