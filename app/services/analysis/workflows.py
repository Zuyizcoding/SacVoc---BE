"""F1/F2: chuẩn bị ảnh, suy luận, hoàn thiện, áp dụng chất lượng và lưu DB."""

import logging
from dataclasses import asdict
from typing import Literal

from google.api_core.exceptions import DeadlineExceeded
from sqlalchemy.orm import Session

from app.core.prompts import get_body_prompts, get_color_prompts
from app.db.crud import save_body_result, save_color_result
from app.schemas.body import BodyApiResponse, BodyGeminiResponse
from app.schemas.color import ColorApiResponse, ColorInferenceResponse
from app.services.gemini_client import GeminiClient, GeminiInvalidResponseError
from app.utils.image_utils import preprocess_image_for_ai

from .errors import (
    AnalysisError,
    AnalysisRejectedError,
    AnalysisUnavailableError,
    GeminiOverloadedError,
)
from .photo_quality import (
    _apply_clothing_soft_fail_adjustments,
    _apply_soft_quality_adjustments,
    _photo_quality_score_value,
    needs_body_quality_hint,
    needs_color_quality_hint,
)
from .photo_types import ProcessedImage
from .prompts import (
    _build_body_best_effort_prompt,
    _build_body_completion_prompt,
    _build_body_medium_quality_hint_prompt,
    _build_color_completion_prompt,
    _build_color_quality_hint_prompt,
)
from .results import (
    _build_demo_body_payload,
    _build_demo_color_payload,
    _color_missing_fields,
    _finalize_color_result,
    merge_nonempty_result,
)

logger = logging.getLogger(__name__)


def _reject_if_error(result: dict, fallback: str) -> None:
    if result.get("error_code") is not None:
        code = str(result["error_code"]).strip().upper() or "ANALYSIS_FAILED"
        raise AnalysisRejectedError(result.get("message", fallback), code)


async def _infer_color(image: ProcessedImage, client: GeminiClient) -> dict:
    system, prompt = get_color_prompts()
    if needs_color_quality_hint(image.quality) or image.quality.soft_low_detail:
        prompt += "\n\n" + _build_color_quality_hint_prompt(image.quality)
    result = await client.analyze(
        image_bytes=image.jpeg_bytes,
        prompt=prompt,
        response_schema=ColorInferenceResponse,
        system_instruction=system,
    )
    _reject_if_error(
        result, "Không thể phân tích ảnh. Vui lòng thử lại với ảnh rõ hơn."
    )
    missing = _color_missing_fields(result)
    if missing:
        repaired = await client.analyze(
            image_bytes=image.jpeg_bytes,
            prompt=_build_color_completion_prompt(missing, result, image.quality),
            response_schema=ColorInferenceResponse,
            system_instruction=system,
        )
        if repaired.get("error_code") is None:
            result = merge_nonempty_result(result, repaired)
    if "sub_season" in _color_missing_fields(result):
        raise AnalysisUnavailableError(
            "AI phản hồi chưa đủ dữ liệu để kết luận mùa sắc đáng tin cậy. Vui lòng thử lại sau vài giây hoặc dùng ảnh selfie rõ hơn.",
            "AI_INCOMPLETE",
        )
    return _finalize_color_result(result, image.quality)


async def _infer_body_pass(
    image: ProcessedImage, client: GeminiClient, prompt: str, *, repair: bool = False
) -> dict:
    """Mỗi lượt có tối đa một lần thử best-effort khi quần áo che dáng."""
    system, _ = get_body_prompts()
    result = await client.analyze(
        image_bytes=image.jpeg_bytes,
        prompt=prompt,
        response_schema=BodyGeminiResponse,
        system_instruction=system,
    )
    if str(result.get("error_code") or "").strip().upper() == "CLOTHING_TOO_WIDE":
        result = await client.analyze(
            image_bytes=image.jpeg_bytes,
            prompt=_build_body_best_effort_prompt(),
            response_schema=BodyGeminiResponse,
            system_instruction=system,
        )
        if result.get("error_code") is not None:
            fallback = (
                "Không thể phân tích dáng người chính xác từ ảnh hiện tại."
                if repair
                else "Không thể phân tích dáng người chính xác do trang phục quá rộng."
            )
            raise AnalysisRejectedError(
                result.get("message", fallback), "CLOTHING_TOO_WIDE"
            )
        result = _apply_clothing_soft_fail_adjustments(result)
    _reject_if_error(
        result, "Không thể phân tích dáng người chính xác từ ảnh hiện tại."
    )
    return result


async def _infer_body(image: ProcessedImage, client: GeminiClient) -> dict:
    _, prompt = get_body_prompts()
    if needs_body_quality_hint(image.quality):
        prompt += "\n\n" + _build_body_medium_quality_hint_prompt(image.quality)
    result = await _infer_body_pass(image, client, prompt)
    parsed = BodyGeminiResponse.model_validate(result)
    missing = parsed.missing_detail_fields()
    if missing:
        repaired = await _infer_body_pass(
            image, client, _build_body_completion_prompt(missing, result), repair=True
        )
        result = merge_nonempty_result(result, repaired)
        parsed = BodyGeminiResponse.model_validate(result)
    if parsed.missing_detail_fields():
        raise AnalysisRejectedError(
            "Ảnh chưa đủ rõ để đưa ra gợi ý dáng người đáng tin cậy. Vui lòng dùng ảnh toàn thân, đứng thẳng và mặc đồ ôm sát hơn.",
            "AI_INCOMPLETE",
        )
    payload = _apply_soft_quality_adjustments(parsed.model_dump(), image.quality)
    if payload.get("confidence") is None:
        payload["confidence"] = round(
            min(0.78, 0.38 + 0.32 * _photo_quality_score_value(image.quality)), 4
        )
    return payload


async def _infer_with_fallback(
    image: ProcessedImage, client: GeminiClient, *, feature: Literal["F1", "F2"]
) -> dict:
    infer = _infer_color if feature == "F1" else _infer_body
    demo = _build_demo_color_payload if feature == "F1" else _build_demo_body_payload
    try:
        result = await infer(image, client)
        result["analysis_source"] = "live"
        return result
    except (GeminiOverloadedError, TimeoutError, DeadlineExceeded):
        logger.warning("%s dùng demo do Gemini quá tải hoặc timeout", feature)
        return demo(image.quality)
    except AnalysisError:
        raise
    except GeminiInvalidResponseError as exc:
        subject = "màu sắc" if feature == "F1" else "dáng người"
        photo = "selfie" if feature == "F1" else "toàn thân"
        raise AnalysisUnavailableError(
            f"AI phản hồi chưa đầy đủ cho phân tích {subject}. Vui lòng thử lại sau vài giây hoặc dùng ảnh {photo} rõ hơn.",
            "AI_INCOMPLETE",
        ) from exc
    except ValueError as exc:
        subject = "màu sắc" if feature == "F1" else "dáng người"
        raise AnalysisUnavailableError(
            f"AI trả về kết quả chưa nhất quán cho phân tích {subject}. Vui lòng thử lại sau vài giây.",
            "AI_INVALID_RESULT",
        ) from exc


async def analyze_color_image(
    image_bytes: bytes, device_id: str, db: Session, client: GeminiClient
) -> ColorApiResponse:
    image = preprocess_image_for_ai(image_bytes, max_kb=1000)
    logger.info("F1 photo quality=%s", asdict(image.quality))
    result = await _infer_with_fallback(image, client, feature="F1")
    save_color_result(
        db=db,
        device_id=device_id,
        sub_season=result["sub_season"],
        primary_season=result["primary_season"],
        confidence=result["confidence"],
        full_result=result,
    )
    return ColorApiResponse.model_validate(result)


async def analyze_body_image(
    image_bytes: bytes, device_id: str, db: Session, client: GeminiClient
) -> BodyApiResponse:
    image = preprocess_image_for_ai(image_bytes, max_kb=1000)
    logger.info("F2 photo quality=%s", asdict(image.quality))
    result = await _infer_with_fallback(image, client, feature="F2")
    save_body_result(
        db=db,
        device_id=device_id,
        shape=result["shape"],
        confidence=float(result["confidence"]),
        full_result=result,
    )
    return BodyApiResponse.model_validate(result)
