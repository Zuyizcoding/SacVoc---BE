"""
Router phân tích ảnh cho Color (F1) + Body (F2).

Tuân thủ:
  - backend-python #1: Router riêng biệt cho /analyze
  - backend-python #2: Depends() inject GeminiClient + DB Session
  - Tính năng #9: try/finally xóa ảnh tạm - KHÔNG lưu ảnh người dùng
  - Tính năng #34: HTTP 503/422/429/400 -> thông báo tiếng Việt rõ ràng
  - Tính năng #15: F1+F2->F3 combined flow - đọc kết quả từ DB

Luồng xử lý:
  1. Nhận ảnh upload (UploadFile)
  2. Validate kích thước (>=400x400px)
  3. Compress (<=1MB JPEG)
  4. Gửi Gemini qua GeminiClient
  5. Override palette từ season_data.py (F1)
  6. Lưu kết quả vào ANALYSIS_HISTORY
  7. finally: Xóa ảnh tạm (dù thành công hay thất bại)
"""
import os
import logging
import tempfile
import json

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Header
from google.api_core.exceptions import DeadlineExceeded
from sqlalchemy.orm import Session

from app.schemas.base import ApiResponse
from app.schemas.color import ColorApiResponse, ColorInferenceResponse
from app.schemas.body import BodyApiResponse, BodyGeminiResponse
from app.core.prompts import get_body_prompts, get_color_prompts
from app.services.gemini_client import (
    GeminiClient,
    GeminiInvalidResponseError,
    get_gemini_client,
)
from app.services.demo_cache import get_demo_body_result, get_demo_color_result
from app.schemas.history import HistoryApiResponse
from app.utils.image_utils import preprocess_image_for_ai
from app.db.database import get_db
from app.db.crud import get_analysis, save_color_result, save_body_result
from app.data.season_data import (
    canonicalize_sub_season,
    get_season_entry,
    get_menswear_recommendation,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyze", tags=["Analysis"])

ERROR_CODE_PHOTO_UNREADABLE = "PHOTO_UNREADABLE"
ERROR_CODE_PHOTO_TOO_SMALL = "PHOTO_TOO_SMALL"
ERROR_CODE_PHOTO_TOO_FAR = "PHOTO_TOO_FAR"
ERROR_CODE_NO_PERSON = "NO_PERSON_DETECTED"
ERROR_CODE_MULTIPLE_PEOPLE = "MULTIPLE_PEOPLE_DETECTED"
ERROR_CODE_CLOTHING_TOO_WIDE = "CLOTHING_TOO_WIDE"
ERROR_CODE_AI_INCOMPLETE = "AI_INCOMPLETE"
ERROR_CODE_AI_OVERLOADED = "AI_OVERLOADED"
ERROR_CODE_AI_INVALID_RESULT = "AI_INVALID_RESULT"
ERROR_CODE_ANALYSIS_FAILED = "ANALYSIS_FAILED"
DEMO_CACHE_COLOR_NOTICE = (
    "Lưu ý: hệ thống đang dùng kết quả dự phòng vì AI tạm thời quá tải, nên đây là gợi ý demo tham khảo."
)
DEMO_CACHE_BODY_NOTICE = (
    "Lưu ý: hệ thống đang dùng kết quả dự phòng vì AI tạm thời quá tải, nên đây là gợi ý demo tham khảo."
)


def _typed_error_detail(message: str, error_code: str) -> dict[str, str]:
    return {"error_code": error_code, "message": message}


def _raise_typed_http(status_code: int, message: str, error_code: str) -> None:
    raise HTTPException(status_code=status_code, detail=_typed_error_detail(message, error_code))


def _api_error_response(message: str, error_code: str) -> ApiResponse[None]:
    return ApiResponse(success=False, data=None, error=message, error_code=error_code)


def _append_notice(text: str, notice: str) -> str:
    base = text.strip()
    if not base:
        return notice
    if notice in base:
        return base
    return f"{base} {notice}"


def _is_ai_overloaded_http(exc: HTTPException) -> bool:
    if exc.status_code == 429:
        return True

    detail = exc.detail
    if isinstance(detail, dict):
        return str(detail.get("error_code") or "").strip().upper() == ERROR_CODE_AI_OVERLOADED
    if isinstance(detail, str):
        lowered = detail.lower()
        return "quá tải" in lowered or "rate limit" in lowered or "free tier" in lowered
    return False


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


def _merge_body_result(base: dict, patch: dict) -> dict:
    """Merge partial F2 payloads, ưu tiên dữ liệu mới khi field có giá trị."""
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def _merge_color_result(base: dict, patch: dict) -> dict:
    """Merge partial F1 payloads, giữ field cũ khi patch vẫn trống."""
    merged = dict(base)
    for key, value in patch.items():
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, str) and not value.strip():
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


def _build_color_quality_hint_prompt(quality_flags: dict) -> str:
    reasons: list[str] = []
    if quality_flags.get("soft_far_subject_for_color"):
        reasons.append("chủ thể trong ảnh chiếm khung hình chưa nhiều")
    if quality_flags.get("soft_low_detail"):
        reasons.append("ảnh hơi mềm hoặc thiếu chi tiết")
    if quality_flags.get("soft_bad_lighting"):
        reasons.append("ánh sáng chưa thật sự tối ưu")
    if quality_flags.get("soft_color_cast"):
        reasons.append("ảnh có thể bị lệch màu nhẹ")

    joined = ", ".join(reasons) if reasons else "ảnh chưa thật sự tối ưu"
    return (
        "<quality_hint>\n"
        f"Ảnh này vẫn có thể phân tích, nhưng {joined}.\n"
        "Hãy ưu tiên phân tích best-effort thay vì từ chối.\n"
        "Nếu độ chắc chắn không cao, hãy hạ confidence ở mức hợp lý: lỗi nhẹ khoảng 0.55-0.75,\n"
        "nhiều vấn đề hơn thì có thể thấp hơn, và nói rõ rằng đây là kết quả tham khảo.\n"
        "Không được từ chối chỉ vì ảnh hơi xa, hơi mềm hoặc ánh sáng chưa hoàn hảo nếu vẫn thấy rõ 1 người.\n"
        "</quality_hint>"
    )


def _build_color_completion_prompt(
    missing_fields: list[str],
    partial_result: dict,
    quality_flags: dict,
) -> str:
    missing_text = ", ".join(missing_fields)
    return (
        "<task>\n"
        "Bạn vừa phân tích seasonal color từ ảnh này nhưng còn thiếu một số field quan trọng.\n"
        f"Hãy phân tích lại cùng ảnh và chỉ tập trung bổ sung: {missing_text}.\n"
        "Giữ nguyên hướng phân tích đã có nếu sub_season hiện tại vẫn hợp lý.\n"
        "Nếu ảnh chưa đủ hoàn hảo, vẫn trả best-effort với confidence thấp thay vì từ chối.\n"
        "secondary_season có thể để trống nếu không chắc, nhưng sub_season phải là 1 trong 12 mùa chuẩn.\n"
        f"Partial result trước đó để tham chiếu: {json.dumps(partial_result, ensure_ascii=False)}\n"
        f"{_build_color_quality_hint_prompt(quality_flags)}\n"
        "</task>"
    )


def _fallback_color_explanation(sub_season: str, quality_flags: dict) -> str:
    explanation = (
        f"Hệ thống nhận diện bạn gần nhất với nhóm {sub_season} dựa trên tổng thể da, tóc và mắt trong ảnh."
    )
    if quality_flags.get("soft_far_subject_for_color") or quality_flags.get("soft_low_detail"):
        return (
            f"{explanation} Tuy vậy ảnh hiện tại chưa thật sự lý tưởng nên kết quả này nên được dùng như tham khảo."
        )
    return explanation


def _fallback_color_advice(result: dict) -> str:
    menswear = result.get("menswear")
    if isinstance(menswear, dict):
        tops = [item for item in menswear.get("tops", []) if isinstance(item, str) and item.strip()]
        bottoms = [
            item for item in menswear.get("bottoms", []) if isinstance(item, str) and item.strip()
        ]
        if tops and bottoms:
            return (
                f"Ưu tiên {tops[0]} cho áo/layer và phối cùng {bottoms[0]} để tổng thể dễ hài hòa hơn."
            )
        if tops:
            return f"Ưu tiên các gam như {tops[0]} cho áo hoặc layer để gương mặt trông sáng và cân bằng hơn."
    return "Ưu tiên các gam màu nằm trong bảng gợi ý và chọn outfit có độ tương phản vừa phải để tôn da hơn."


def _build_body_completion_prompt(missing_fields: list[str], partial_result: dict) -> str:
    """Prompt ngắn để yêu cầu Gemini chỉ hoàn thiện phần còn thiếu."""
    missing_text = ", ".join(missing_fields)
    return (
        "<task>\n"
        "Bạn vừa phân tích body shape từ ảnh này nhưng còn thiếu một số field bắt buộc.\n"
        f"Hãy phân tích lại cùng ảnh và trả JSON hoàn chỉnh, tập trung bổ sung: {missing_text}.\n"
        "Chỉ điền measurements khi thực sự quan sát được tỷ lệ vai/hông/eo từ ảnh;\n"
        "không được suy diễn measurements chỉ từ shape name.\n"
        "Nếu ảnh không đủ rõ để kết luận chắc chắn, ưu tiên vẫn trả kết quả với confidence thấp,\n"
        "chỉ trả JSON lỗi CLOTHING_TOO_WIDE khi trang phục che dáng quá nặng và không thể suy luận tỷ lệ cơ bản.\n"
        "Không được bịa mẹo mặc đồ chung chung hoặc cố làm kết quả trông chắc chắn hơn thực tế.\n"
        f"Partial result trước đó để tham chiếu: {json.dumps(partial_result, ensure_ascii=False)}\n"
        "</task>"
    )


def _build_body_best_effort_prompt() -> str:
    """Prompt fail-safe: force best-effort analysis for borderline wide clothing."""
    return (
        "<task>\n"
        "Ảnh này có thể mặc đồ hơi rộng, nhưng hãy ưu tiên phân tích best-effort.\n"
        "Yêu cầu:\n"
        "1) VẪN trả đầy đủ JSON body shape (shape, confidence, measurements, styling_tips, avoid, explanation).\n"
        "2) Đặt confidence thấp (0.25-0.45) nếu độ chắc chắn không cao.\n"
        "3) Trong explanation phải nêu rõ là trang phục hơi rộng nên kết quả chỉ tham khảo.\n"
        "4) CHỈ trả lỗi CLOTHING_TOO_WIDE khi bị che dáng cực nặng: không nhìn thấy tương quan vai/eo/hông một cách tối thiểu.\n"
        "5) Không bịa dữ liệu, không dùng câu chung chung.\n"
        "</task>"
    )


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


def _should_return_color_close_up_guidance(quality_flags: dict) -> bool:
    return bool(quality_flags.get("soft_far_subject_for_color"))


def _color_close_up_guidance_message(_quality_flags: dict) -> str:
    return (
        "Ảnh hiện tại chụp hơi xa nên phù hợp để phân tích dáng người hơn là phân tích màu sắc. "
        "Để phân tích màu chính xác hơn, hãy dùng ảnh selfie gần hơn, thấy rõ khuôn mặt và vùng cổ, "
        "ánh sáng đều và không đeo kính râm/phụ kiện che mặt."
    )


def _should_bias_body_best_effort(quality_flags: dict) -> bool:
    return (
        bool(quality_flags.get("soft_low_detail"))
        and not bool(quality_flags.get("soft_low_resolution"))
        and not bool(quality_flags.get("soft_bad_lighting"))
        and not bool(quality_flags.get("soft_color_cast"))
    )


def _build_body_medium_quality_hint_prompt(quality_flags: dict) -> str:
    detail_score = quality_flags.get("detail_score")
    return (
        "<quality_hint>\n"
        "Ảnh này là ảnh toàn thân mức trung bình: vẫn nhìn thấy silhouette tổng thể, "
        "nhưng chi tiết không quá cao.\n"
        f"Detail score tham khảo: {detail_score}.\n"
        "Hãy ưu tiên phân tích best-effort dựa trên tương quan vai/eo/hông và form tổng thể.\n"
        "Không được từ chối chỉ vì khuôn mặt nhỏ, ảnh hơi mềm hoặc áo thun hơi rộng.\n"
        "Chỉ trả CLOTHING_TOO_WIDE khi thực sự không nhìn thấy tương quan vai/eo/hông ở mức tối thiểu.\n"
        "Nếu độ chắc chắn thấp, hạ confidence xuống 0.25-0.45 và ghi rõ là kết quả tham khảo.\n"
        "</quality_hint>"
    )


def _soft_quality_warning_message(quality_flags: dict) -> str:
    reasons: list[str] = []
    if quality_flags.get("soft_far_subject_for_color"):
        reasons.append("chủ thể trong ảnh hơi nhỏ hoặc hơi xa")
    if quality_flags.get("soft_low_resolution"):
        reasons.append("độ phân giải còn thấp")
    if quality_flags.get("soft_low_detail"):
        reasons.append("ảnh thiếu chi tiết")
    if quality_flags.get("soft_bad_lighting"):
        reasons.append("ánh sáng chưa ổn định")
    if quality_flags.get("soft_color_cast"):
        reasons.append("ảnh có lệch màu")

    joined = ", ".join(reasons) if reasons else "chất lượng ảnh chưa tối ưu"
    return (
        f"Ảnh hiện tại có {joined}, nên kết quả chỉ nên dùng để tham khảo. "
        "Bạn có thể chụp ảnh sáng hơn, cân bằng màu hơn và rõ hơn để tăng độ chính xác."
    )


def _photo_quality_score_value(quality_flags: dict) -> float:
    raw_score = quality_flags.get("photo_quality_score")
    if isinstance(raw_score, (int, float)):
        return max(0.0, min(float(raw_score), 1.0))
    return 0.75


def _photo_quality_band_value(quality_flags: dict) -> str:
    raw_band = str(quality_flags.get("photo_quality_band") or "").strip().lower()
    if raw_band in {"excellent", "good", "usable", "poor"}:
        return raw_band
    score = _photo_quality_score_value(quality_flags)
    if score >= 0.85:
        return "excellent"
    if score >= 0.72:
        return "good"
    if score >= 0.58:
        return "usable"
    return "poor"


def _soft_quality_confidence_cap(quality_flags: dict) -> float | None:
    far_subject = bool(quality_flags.get("soft_far_subject_for_color"))
    low_resolution = bool(quality_flags.get("soft_low_resolution"))
    low_detail = bool(quality_flags.get("soft_low_detail"))
    bad_lighting = bool(quality_flags.get("soft_bad_lighting"))
    color_cast = bool(quality_flags.get("soft_color_cast"))
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

    soft_issue_count = sum(1 for flag in (low_resolution, low_detail, bad_lighting, color_cast) if flag)
    if soft_issue_count >= 2:
        cap = min(cap, 0.64)
    if soft_issue_count >= 3:
        cap = min(cap, 0.58)
    if far_subject and soft_issue_count >= 1:
        cap = min(cap, 0.52)

    return cap


def _attach_photo_quality_metadata(result: dict, quality_flags: dict) -> dict:
    patched = dict(result)
    patched["photo_quality_score"] = round(_photo_quality_score_value(quality_flags), 4)
    patched["photo_quality_band"] = _photo_quality_band_value(quality_flags)
    return patched


def _apply_soft_quality_adjustments(result: dict, quality_flags: dict) -> dict:
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


def _finalize_color_result(result: dict, quality_flags: dict) -> dict:
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
        patched["explanation"] = _fallback_color_explanation(patched["sub_season"], quality_flags)

    if not str(patched.get("advice") or "").strip():
        patched["advice"] = _fallback_color_advice(patched)

    return _apply_soft_quality_adjustments(patched, quality_flags)


def _build_demo_color_payload(quality_flags: dict) -> dict:
    payload = _finalize_color_result(get_demo_color_result(), quality_flags)
    payload["analysis_source"] = "degraded_demo"
    payload["explanation"] = _append_notice(
        str(payload.get("explanation") or "").strip(),
        DEMO_CACHE_COLOR_NOTICE,
    )
    payload["advice"] = _append_notice(
        str(payload.get("advice") or "").strip(),
        "Lưu ý: hệ thống đang dùng chế độ dự phòng do AI tạm thời quá tải.",
    )
    return payload


def _build_demo_body_payload(quality_flags: dict) -> dict:
    payload = BodyGeminiResponse.model_validate(get_demo_body_result()).model_dump()
    payload = _apply_soft_quality_adjustments(payload, quality_flags)
    payload["analysis_source"] = "degraded_demo"
    if payload.get("confidence") is None:
        payload["confidence"] = round(
            min(0.72, 0.38 + 0.28 * _photo_quality_score_value(quality_flags)),
            4,
        )
    payload["explanation"] = _append_notice(
        str(payload.get("explanation") or "").strip(),
        DEMO_CACHE_BODY_NOTICE,
    )
    return payload


def _demo_color_response(
    *,
    db: Session,
    device_id: str,
    quality_flags: dict,
) -> ApiResponse[ColorApiResponse]:
    result = _build_demo_color_payload(quality_flags)
    save_color_result(
        db=db,
        device_id=device_id,
        sub_season=result["sub_season"],
        primary_season=result["primary_season"],
        confidence=result["confidence"],
        full_result=result,
    )
    return ApiResponse(
        success=True,
        data=ColorApiResponse.model_validate(result),
    )


def _demo_body_response(
    *,
    db: Session,
    device_id: str,
    quality_flags: dict,
) -> ApiResponse[BodyApiResponse]:
    result = _build_demo_body_payload(quality_flags)
    parsed_result = BodyGeminiResponse.model_validate(result)
    save_body_result(
        db=db,
        device_id=device_id,
        shape=parsed_result.shape,
        confidence=float(result["confidence"]),
        full_result=result,
    )
    return ApiResponse(
        success=True,
        data=BodyApiResponse.model_validate(result),
    )


# ----------------------------------------------------------------------
# POST /api/v1/analyze/color (Tính năng #16, #17, #18, #19)
# ----------------------------------------------------------------------
@router.post(
    "/color",
    response_model=ApiResponse[ColorApiResponse],
    summary="F1 - Phân tích màu sắc cá nhân",
    description="Nhận ảnh selfie, trả về 12-season + palette + makeup guide.",
)
async def analyze_color(
    file: UploadFile = File(..., description="Ảnh selfie JPEG/PNG"),
    x_device_id: str = Header(
        ..., description="SHA256 device ID (Tính năng #3)"
    ),
    client: GeminiClient = Depends(get_gemini_client),
    db: Session = Depends(get_db),
) -> ApiResponse[ColorApiResponse]:
    """Phân tích màu sắc cá nhân từ ảnh selfie.

    Flow:
        Upload -> Validate -> Compress -> Gemini -> Override palette -> Response
    """
    tmp_path: str | None = None
    quality_flags: dict = {}

    try:
        # --- Bước 1: Đọc file upload ---
        file_bytes = await file.read()
        logger.info(
            "Color analysis request | device=%s | size=%d bytes",
            x_device_id[:8],
            len(file_bytes),
        )

        compressed, quality_flags = preprocess_image_for_ai(
            file_bytes,
            max_kb=1000,
        )
        logger.info("F1 quality_flags=%s", json.dumps(quality_flags, ensure_ascii=False))

        # --- Bước 4: Lưu tạm (để log/debug nếu cần) ---
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as tmp:
            tmp.write(compressed)
            tmp_path = tmp.name

        # --- Bước 5: Gọi Gemini (Tính năng #10, #11, #12) ---
        system_instruction_f1, prompt_f1 = get_color_prompts()
        if _should_return_color_close_up_guidance(quality_flags) or bool(
            quality_flags.get("soft_low_detail")
        ):
            prompt_f1 = f"{prompt_f1}\n\n{_build_color_quality_hint_prompt(quality_flags)}"

        result = await client.analyze(
            image_bytes=compressed,
            prompt=prompt_f1,
            response_schema=ColorInferenceResponse,
            system_instruction=system_instruction_f1,
        )

        # If Gemini returns a structured error, propagate as an API-level error.
        if result.get("error_code") is not None:
            return _api_error_response(
                result.get(
                    "message",
                    "Không thể phân tích ảnh. Vui lòng thử lại với ảnh rõ hơn.",
                ),
                str(result.get("error_code")).strip().upper() or ERROR_CODE_ANALYSIS_FAILED,
            )

        missing_fields = _color_missing_fields(result)
        if missing_fields:
            logger.warning(
                "F1 thiếu field sau lượt đầu: %s | device=%s",
                missing_fields,
                x_device_id[:8],
            )
            repair_prompt = _build_color_completion_prompt(
                missing_fields,
                result,
                quality_flags,
            )
            repaired = await client.analyze(
                image_bytes=compressed,
                prompt=repair_prompt,
                response_schema=ColorInferenceResponse,
                system_instruction=system_instruction_f1,
            )
            if repaired.get("error_code") is None:
                result = _merge_color_result(result, repaired)
            else:
                logger.info(
                    "F1 repair returned structured error %s; keep first pass if still usable",
                    repaired.get("error_code"),
                )

        missing_fields = _color_missing_fields(result)
        if missing_fields and "sub_season" in missing_fields:
            _raise_typed_http(
                503,
                (
                    "AI phản hồi chưa đủ dữ liệu để kết luận mùa sắc đáng tin cậy. "
                    "Vui lòng thử lại sau vài giây hoặc dùng ảnh selfie rõ hơn."
                ),
                ERROR_CODE_AI_INCOMPLETE,
            )

        # --- Bước 6: Canonicalize + enrich F1 từ season_data.py ---
        result = _finalize_color_result(result, quality_flags)
        result["analysis_source"] = "live"

        # --- Bước 7: Lưu vào ANALYSIS_HISTORY ---
        save_color_result(
            db=db,
            device_id=x_device_id,
            sub_season=result["sub_season"],
            primary_season=result["primary_season"],
            confidence=result["confidence"],
            full_result=result,
        )

        return ApiResponse(
            success=True,
            data=ColorApiResponse.model_validate(result),
        )

    # -------------------- Error Handling (Tính năng #34) --------------------
    except HTTPException as exc:
        if _is_ai_overloaded_http(exc):
            logger.warning("F1 fallback to demo cache due to AI overload | device=%s", x_device_id[:8])
            return _demo_color_response(
                db=db,
                device_id=x_device_id,
                quality_flags=quality_flags,
            )
        raise  # 422 từ validate_image - giữ nguyên

    except TimeoutError:
        logger.error("Gemini timeout cho F1 -> fallback demo cache")
        return _demo_color_response(
            db=db,
            device_id=x_device_id,
            quality_flags=quality_flags,
        )

    except DeadlineExceeded:
        logger.error("Gemini timeout (DeadlineExceeded) cho F1 -> fallback demo cache")
        return _demo_color_response(
            db=db,
            device_id=x_device_id,
            quality_flags=quality_flags,
        )

    except GeminiInvalidResponseError as exc:
        logger.warning("⚠️ Gemini returned incomplete F1 response: %s", str(exc))
        _raise_typed_http(
            503,
            (
                "AI phản hồi chưa đầy đủ cho phân tích màu sắc. "
                "Vui lòng thử lại sau vài giây hoặc dùng ảnh selfie rõ hơn."
            ),
            ERROR_CODE_AI_INCOMPLETE,
        )

    except ValueError as exc:
        logger.error("Gemini JSON parse error: %s", str(exc))
        _raise_typed_http(
            503,
            (
                "AI trả về kết quả chưa nhất quán cho phân tích màu sắc. "
                "Vui lòng thử lại sau vài giây."
            ),
            ERROR_CODE_AI_INVALID_RESULT,
        )

    except Exception as exc:
        logger.exception("Lỗi không xác định trong F1: %s", str(exc))
        _raise_typed_http(503, "Hệ thống gặp sự cố. Vui lòng thử lại sau.", ERROR_CODE_ANALYSIS_FAILED)

    # ---------------------- Photo Cleanup (Tính năng #9) ---------------------
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Đã xóa ảnh tạm: %s", tmp_path)


# ----------------------------------------------------------------------
# POST /api/v1/analyze/body (Tính năng #23, #24, #25, #26)
# ----------------------------------------------------------------------
@router.post(
    "/body",
    response_model=ApiResponse[BodyApiResponse],
    summary="F2 - Phân tích dáng người",
    description="Nhận ảnh toàn thân, trả về body shape + styling tips.",
)
async def analyze_body(
    file: UploadFile = File(..., description="Ảnh toàn thân JPEG/PNG"),
    x_device_id: str = Header(
        ..., description="SHA256 device ID (Tính năng #3)"
    ),
    client: GeminiClient = Depends(get_gemini_client),
    db: Session = Depends(get_db),
) -> ApiResponse[BodyApiResponse]:
    """Phân tích dáng người từ ảnh toàn thân.

    Flow tương tự F1 - copy pattern try/finally.
    """
    tmp_path: str | None = None
    quality_flags: dict = {}

    try:
        file_bytes = await file.read()
        logger.info(
            "Body analysis request | device=%s | size=%d bytes",
            x_device_id[:8],
            len(file_bytes),
        )

        compressed, quality_flags = preprocess_image_for_ai(
            file_bytes,
            max_kb=1000,
        )
        logger.info("F2 quality_flags=%s", json.dumps(quality_flags, ensure_ascii=False))

        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as tmp:
            tmp.write(compressed)
            tmp_path = tmp.name

        system_instruction_f2, prompt_f2 = get_body_prompts()
        if _should_bias_body_best_effort(quality_flags):
            prompt_f2 = f"{prompt_f2}\n\n{_build_body_medium_quality_hint_prompt(quality_flags)}"

        result = await client.analyze(
            image_bytes=compressed,
            prompt=prompt_f2,
            response_schema=BodyGeminiResponse,
            system_instruction=system_instruction_f2,
        )

        # For wide-clothing edge cases, run one best-effort retry before hard fail.
        if result.get("error_code") is not None:
            if str(result.get("error_code")).strip().upper() == "CLOTHING_TOO_WIDE":
                logger.info(
                    "F2 CLOTHING_TOO_WIDE on first pass -> run best-effort retry | device=%s",
                    x_device_id[:8],
                )
                retry_prompt = _build_body_best_effort_prompt()
                retry_result = await client.analyze(
                    image_bytes=compressed,
                    prompt=retry_prompt,
                    response_schema=BodyGeminiResponse,
                    system_instruction=system_instruction_f2,
                )
                if retry_result.get("error_code") is None:
                    result = _apply_clothing_soft_fail_adjustments(retry_result)
                else:
                    return ApiResponse(
                        success=False,
                        data=None,
                        error=retry_result.get(
                            "message",
                            "Không thể phân tích dáng người chính xác do trang phục quá rộng.",
                        ),
                        error_code=ERROR_CODE_CLOTHING_TOO_WIDE,
                    )
            else:
                return ApiResponse(
                    success=False,
                    data=None,
                    error=result.get(
                        "message",
                        "Không thể phân tích dáng người chính xác từ ảnh hiện tại.",
                    ),
                    error_code=str(result.get("error_code")).strip().upper() or ERROR_CODE_ANALYSIS_FAILED,
                )

        parsed_result = BodyGeminiResponse.model_validate(result)
        missing_fields = parsed_result.missing_detail_fields()
        if missing_fields:
            logger.warning(
                "F2 thiếu field grounded sau lượt đầu: %s | device=%s",
                missing_fields,
                x_device_id[:8],
            )
            repair_prompt = _build_body_completion_prompt(missing_fields, result)
            repaired = await client.analyze(
                image_bytes=compressed,
                prompt=repair_prompt,
                response_schema=BodyGeminiResponse,
                system_instruction=system_instruction_f2,
            )

            if repaired.get("error_code") is not None:
                if str(repaired.get("error_code")).strip().upper() == "CLOTHING_TOO_WIDE":
                    logger.info(
                        "F2 CLOTHING_TOO_WIDE on repair pass -> run best-effort retry | device=%s",
                        x_device_id[:8],
                    )
                    retry_prompt = _build_body_best_effort_prompt()
                    retry_result = await client.analyze(
                        image_bytes=compressed,
                        prompt=retry_prompt,
                        response_schema=BodyGeminiResponse,
                        system_instruction=system_instruction_f2,
                    )
                    if retry_result.get("error_code") is None:
                        repaired = _apply_clothing_soft_fail_adjustments(retry_result)
                    else:
                        return ApiResponse(
                            success=False,
                            data=None,
                            error=retry_result.get(
                                "message",
                                "Không thể phân tích dáng người chính xác từ ảnh hiện tại.",
                            ),
                            error_code=ERROR_CODE_CLOTHING_TOO_WIDE,
                        )
                else:
                    return ApiResponse(
                        success=False,
                        data=None,
                        error=repaired.get(
                            "message",
                            "Không thể phân tích dáng người chính xác từ ảnh hiện tại.",
                        ),
                        error_code=str(repaired.get("error_code")).strip().upper() or ERROR_CODE_ANALYSIS_FAILED,
                    )

            result = _merge_body_result(result, repaired)
            parsed_result = BodyGeminiResponse.model_validate(result)
            missing_fields = parsed_result.missing_detail_fields()

        if missing_fields:
            logger.warning(
                "F2 vẫn thiếu field sau repair pass: %s | device=%s",
                missing_fields,
                x_device_id[:8],
            )
            return ApiResponse(
                success=False,
                data=None,
                error=(
                    "Ảnh chưa đủ rõ để đưa ra gợi ý dáng người đáng tin cậy. "
                    "Vui lòng dùng ảnh toàn thân, đứng thẳng và mặc đồ ôm sát hơn."
                ),
                error_code=ERROR_CODE_AI_INCOMPLETE,
            )

        response_payload = parsed_result.model_dump()
        response_payload = _apply_soft_quality_adjustments(
            response_payload,
            quality_flags,
        )
        response_payload["analysis_source"] = "live"
        if response_payload.get("confidence") is None:
            response_payload["confidence"] = round(
                min(0.78, 0.38 + 0.32 * _photo_quality_score_value(quality_flags)),
                4,
            )

        save_body_result(
            db=db,
            device_id=x_device_id,
            shape=parsed_result.shape,
            confidence=float(response_payload["confidence"]),
            full_result=response_payload,
        )

        return ApiResponse(
            success=True,
            data=BodyApiResponse.model_validate(response_payload),
        )

    except HTTPException as exc:
        if _is_ai_overloaded_http(exc):
            logger.warning("F2 fallback to demo cache due to AI overload | device=%s", x_device_id[:8])
            return _demo_body_response(
                db=db,
                device_id=x_device_id,
                quality_flags=quality_flags,
            )
        raise

    except TimeoutError:
        logger.error("Gemini timeout cho F2 -> fallback demo cache")
        return _demo_body_response(
            db=db,
            device_id=x_device_id,
            quality_flags=quality_flags,
        )

    except DeadlineExceeded:
        logger.error("Gemini timeout (DeadlineExceeded) cho F2 -> fallback demo cache")
        return _demo_body_response(
            db=db,
            device_id=x_device_id,
            quality_flags=quality_flags,
        )

    except GeminiInvalidResponseError as exc:
        logger.warning("⚠️ Gemini returned incomplete F2 response: %s", str(exc))
        _raise_typed_http(
            503,
            (
                "AI phản hồi chưa đầy đủ cho phân tích dáng người. "
                "Vui lòng thử lại sau vài giây hoặc dùng ảnh toàn thân rõ hơn."
            ),
            ERROR_CODE_AI_INCOMPLETE,
        )

    except ValueError as exc:
        logger.error("Body JSON parse error: %s", str(exc))
        _raise_typed_http(
            503,
            (
                "AI trả về kết quả chưa nhất quán cho phân tích dáng người. "
                "Vui lòng thử lại sau vài giây."
            ),
            ERROR_CODE_AI_INVALID_RESULT,
        )

    except Exception as exc:
        logger.exception("Lỗi không xác định trong F2: %s", str(exc))
        _raise_typed_http(503, "Hệ thống gặp sự cố. Vui lòng thử lại sau.", ERROR_CODE_ANALYSIS_FAILED)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Đã xóa ảnh tạm: %s", tmp_path)


# ----------------------------------------------------------------------
# GET /api/v1/analyze/history (SHOULD - Xem lại kết quả)
# ----------------------------------------------------------------------
@router.get(
    "/history",
    response_model=ApiResponse[HistoryApiResponse],
    summary="Xem lại kết quả phân tích F1 + F2",
    description="Đọc kết quả phân tích đã lưu trong AnalysisHistory.",
)
async def get_analysis_history(
    x_device_id: str = Header(
        ..., description="SHA256 device ID (Tính năng #3)"
    ),
    db: Session = Depends(get_db),
) -> ApiResponse[HistoryApiResponse]:
    """Trả về lịch sử F1 + F2 nếu đã phân tích.

    SHOULD: Màn "Xem lại kết quả" đọc *_result_json từ AnalysisHistory.
    """
    history = get_analysis(db, x_device_id)

    if history is None:
        return ApiResponse(success=True, data=HistoryApiResponse())

    color: ColorApiResponse | None = None
    body: BodyApiResponse | None = None

    if history.color_result_json:
        color_json = json.loads(history.color_result_json)
        color = ColorApiResponse.model_validate(color_json)

    if history.body_result_json:
        body_json = json.loads(history.body_result_json)
        body = BodyApiResponse.model_validate(body_json)

    return ApiResponse(
        success=True,
        data=HistoryApiResponse(color=color, body=body),
    )

